from __future__ import  annotations
import os
import datetime
import torch
import torch.nn as nn
import numpy as np
# import time
# import random
import kornia as K
import matplotlib.pyplot as plt

import torch.nn.functional as F
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.transforms import rotation_6d_to_matrix

from configs.config import ModelConfig, Config
from models.encoder import ResNetEncoder
from models.gaussian_pointclouds import gaussian_unit_sphere, GPCParams, GaussianPointclouds
from models.vitae import ViTAE
from rendering.renderer import GaussRenderer
from utils.util import crop_and_resize_from_keypoints
from utils.nn import count_parameters, set_requires_grad

from models.invresnet import InvResNet

from utils import log
from models.dino import DINOFusionSimple, get_pretrained_dinov2
from visualization.vis import show_image, make_grid


def normalize_identity(ft_id):
    # return F.normalize(ft_id.reshape(ft_id.shape[0], -1)).reshape(ft_id.shape)
    return ft_id


class NeuralHeadNet(nn.Module):

    def __init__(
            self,
            params: ModelConfig,
            device='cuda',
            train=False,
            tto=False,
            train_cfg: Config | None = None,
    ):
        super().__init__()
        self.device = device
        self.is_main_process = True if train_cfg is None else train_cfg.is_main_process

        self.params = params
        self.tto = tto

        self.training = train
        if train:
            assert train_cfg is not None, "Training mode (train==True) requires train_cfg parameter to be specified!"
        self.train_cfg = train_cfg

        theta_dims = params.feature_map_size
        phi_dims = params.feature_map_size

        self.base_map = gaussian_unit_sphere(
            theta_dims,
            phi_dims,
            radius=params.base_radius,
            sh_degree=params.color_sh_degree,
            k_phi=params.k_phi,
            k_theta=params.k_theta,
            azimuth_range=params.azimuth_range,
            opacity=params.opacity,
        ).to(device)

        if params.polar:
            self.base_map = self.base_map[2:]
            self.base_map[0] = params.base_radius

        self.dim_coarse = 6
        dim_cam = 9   # 6d rotation (a x b) + 3d translation

        self.gpc_params = GPCParams(
            params.color_sh_degree,
            params.polar,
            k_phi=params.k_phi,
            k_theta=params.k_theta
        )

        self.pos = torch.nn.Parameter(torch.zeros(1, 3), requires_grad=self.params.opt_head_pos)
        self.scale = torch.nn.Parameter(torch.zeros(1, 3), requires_grad=self.params.opt_head_pos)

        gaussian_dims = self.gpc_params.num_dims()
        self.scale_factors = torch.ones((1, gaussian_dims, 1, 1)).to(device)

        for layer in self.gpc_params.layers:
            name = layer[0]
            self.scale_factors[:, self.gpc_params.channels[name]] = self.gpc_params.scale_factors[name]

        self.dino = get_pretrained_dinov2(arch=params.dino_arch)
        output_dim_dino = self.dino.blocks[0].attn.qkv.in_features

        img_out_dims = 128
        self.enc_img = ResNetEncoder(
            3,
            img_out_dims,
            # planes=[64, 128, 256],
            # num_blocks=[1, 1, 1],
            # strides=[2, 2, 2],
            planes=[64, 128],
            # planes=[128, 256],
            num_blocks=[1, 1],
            strides=[2, 2],
            flatten=False,
            hidden_dim=None,
        )

        self.dino_fusion = DINOFusionSimple(
            output_dim_dino,
            output_dim=None,
            with_images=params.with_images,
        )
        self.dino_fusion_expr = DINOFusionSimple(
            output_dim_dino,
            output_dim=params.enc_emb_dim,
            with_images=False,
        )

        transformer_in_channels = output_dim_dino * 4 + (img_out_dims if params.with_images else 0)
        self.transformer = ViTAE(
            # in_channels=params.enc_emb_dim,
            in_channels=transformer_in_channels,
            out_channels=self.params.channels_dec,
            image_size=params.image_size//14,
            patch_size=1,
            num_patches_x=self.params.num_patches,
            num_patches_y=self.params.num_patches,
            num_classes=1,
            style_dim=params.dim_expr,
            emb_dropout=self.params.emb_dropout,
            **self.params.vit_params
        )

        self.P_id = InvResNet(
            input_dims=self.params.channels_dec,
            num_blocks=[params.blocks_dec] * params.levels_dec,
            output_size=theta_dims,
            output_channels=gaussian_dims,
            hidden_dim=None,
            upsampling_mode=params.upsample
        )

        #
        # Setup expression enc/dec
        #

        self.enc_expr = ResNetEncoder(
            # input_dim=output_dim_dino * 4 + img_out_dims,
            input_dim=params.enc_emb_dim,
            output_dim=params.dim_expr,
            num_blocks=[2, 2],
            strides=[2, 2],
            flatten=True,
            planes=[256, 256],
            hidden_dim=256
        )
        self.blocks_to_take_id = [2, 5, 8, 11]
        # self.blocks_to_take_expr = [8, 9, 10, 11]
        self.blocks_to_take_expr = [2, 5, 8, 11]

        self.expr_augment = K.augmentation.container.AugmentationSequential(
            K.augmentation.RandomBrightness((0.7, 1.4), p=0.5, same_on_batch=False),
            K.augmentation.RandomPlanckianJitter(mode='blackbody', p=0.5, same_on_batch=False),
            # K.augmentation.RandomHorizontalFlip(p=0.1, same_on_batch=False),
            K.augmentation.RandomRotation(degrees=20, p=0.25, same_on_batch=False),
            # K.augmentation.RandomResizedCrop(size=(280, 280), scale=(3., 3.), ratio=(0.8, 1.), p=1.)
            # K.augmentation.RandomResizedCrop(size=(params.image_size, params.image_size),
            #                                  scale=(0.5, 1.0), ratio=(0.8, 1.0), p=1.)
        )
        self.zoom_in = K.augmentation.container.AugmentationSequential(
            K.augmentation.CenterCrop(size=(int(params.image_size*0.75), int(params.image_size*0.75)), p=1.),
            K.augmentation.Resize(size=(params.image_size, params.image_size), p=1.)
        )
        self.zoom_in_outputs = K.augmentation.container.AugmentationSequential(
            K.augmentation.CenterCrop(size=(int(params.image_size*0.75), int(params.image_size*0.75)), p=1.),
            K.augmentation.Resize(size=(train_cfg.render_size, train_cfg.render_size), p=1.)
        )

        if self.is_main_process:
            log.info("Params DINO: {:,}".format(count_parameters(self.dino)))
            log.info("Params DINOFusion: {:,}".format(count_parameters(self.dino_fusion)))
            log.info("Params DINOFusion expr: {:,}".format(count_parameters(self.dino_fusion_expr)))
            log.info("Params enc_expr: {:,}".format(count_parameters(self.enc_expr)))
            log.info("Params transformer: {:,}".format(count_parameters(self.transformer)))
            log.info("Params decoder: {:,}".format(count_parameters(self.P_id)))

    def get_dino_outputs(self, images):
        with torch.set_grad_enabled(self.training and self.train_cfg.finetune_dino):
            # images_for_dino = K.geometry.resize(images, (self.params.num_patches * 14, self.params.num_patches * 14), )
            dino_outputs = self.dino.get_intermediate_layers(images, n=range(12))
        return dino_outputs

    def encode_identity(self, images, return_dino=True):
        dino_outputs = self.get_dino_outputs(images)
        with torch.set_grad_enabled(self.training):
            img_features = self.enc_img(images) if self.params.with_images else None
            ft_id = self.dino_fusion(
                [dino_outputs[i] for i in self.blocks_to_take_id],
                img_features=img_features,
                output_size=self.params.num_patches
            )
        embeddings = dict(
            ft_img=img_features,
            ft_id=ft_id,
        )
        if return_dino:
            return embeddings, dino_outputs
        else:
            return embeddings

    def encode_expressions(self, images, dino_outputs=None, keypoints=None):
        with torch.set_grad_enabled(self.training):
            if dino_outputs is None:
                if keypoints is not None:
                    B, C, H, W = images.shape
                    assert keypoints.shape[0] == images.shape[0], "images and keypoints need to be same shape"
                    kp2d = keypoints[..., :2] * torch.tensor([W, H]).to(self.device)
                    images = crop_and_resize_from_keypoints(images, kp2d, output_size=(H, W))
                else:
                    images = self.zoom_in(images)
                dino_outputs = self.get_dino_outputs(images)

            encoder_inputs = self.dino_fusion_expr(
                [dino_outputs[i] for i in self.blocks_to_take_expr],
            )
            ft_expr = F.normalize(self.enc_expr(encoder_inputs))

        return dict(ft_expr=ft_expr)

    def decode(self, embeddings, ft_av=None):
        B = embeddings['ft_id'].shape[0]

        #
        # get position and scale of shape prior (sphere)
        #

        base_map = self.base_map.unsqueeze(0).repeat(B, 1, 1, 1)

        if self.params.opt_head_pos and not self.params.global_head_pos:
            pos = torch.tanh(embeddings['ft_coarse'][:, :3]) * 0.25
            scales = torch.sigmoid(embeddings['ft_coarse'][:, 3:6]) + 1.0
        else:
            pos = torch.tanh(self.pos).repeat(B, 1)
            scales = torch.sigmoid(self.scale).repeat(B, 1)

            # These magic offsets are added to produce similar values as the
            # (legacy) head pose optimization above.
            pos[:, 1] = pos[:, 1] + 0.035
            pos[:, 2] = pos[:, 2] - 0.153
            scales[:, 0] = scales[:, 0] + 1.1 - 0.5
            scales[:, 1] = scales[:, 1] + 1.73 - 0.5
            scales[:, 2] = scales[:, 2] + 1.37 - 0.5

        pos[:, 0] = pos[:, 0] + self.params.base_pos_x
        pos[:, 0] = pos[:, 0] * 0  # don't move head horizontally (fix x at zero)
        pos[:, 1] = pos[:, 1] + self.params.base_pos_y
        pos[:, 2] = pos[:, 2] + self.params.base_pos_z

        #
        # decode Gaussians
        #

        with torch.set_grad_enabled(self.tto or self.training):
            if ft_av is None:
                ft_avatar = self.transformer(embeddings['ft_id'], embeddings['ft_expr'])
            else:
                ft_avatar = ft_av
            if self.train_cfg.grow:
                ft_avatar = ft_avatar.detach()
            gaussian_map_id = self.P_id(ft_avatar) * self.scale_factors

        pointclouds = GaussianPointclouds(
            features=base_map + gaussian_map_id,
            pos=pos,
            scales=scales,
            params=self.gpc_params
        )

        embeddings['ft_av'] = ft_avatar

        results = dict(
            embeddings=embeddings,
            base_maps=base_map,
            feature_maps=gaussian_map_id,
            pos=pos,
            scales=scales,
            pointclouds=pointclouds,
        )

        return results

    def cross_attention_parameter_loss(self) -> torch.Tensor:
        """Return an L2 penalty for the cross-attention residual gates.

        The gates are scalar parameters, one per transformer attention layer.
        Keeping this helper on the unwrapped network lets the training loop
        use it transparently with Accelerate/DDP.
        """
        gammas = [layer[0].gamma for layer in self.transformer.transformer.layers]
        if not gammas:
            return self.base_map.new_zeros(())
        return torch.stack([gamma.square().mean() for gamma in gammas]).mean()

    def encode(self, x: torch.Tensor, keypoints: torch.Tensor, pred_expr = False) -> dict:
        embeddings, dino_outputs = self.encode_identity(x)
        dino_outputs = None
        embeddings['ft_expr'] = self.encode_expressions(x, dino_outputs, keypoints=keypoints)['ft_expr'] if pred_expr else None
        return embeddings

    def predict_canonical_maps(
            self,
            identity_images_list: list[torch.Tensor],
            expression_features: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Predict Gaussian offset maps while holding expression fixed."""
        if not identity_images_list:
            raise ValueError("Canonical-map prediction requires at least one identity view")

        batch_size = len(identity_images_list[0])
        if any(len(images) != batch_size for images in identity_images_list):
            raise ValueError("All identity views must have the same batch size")
        if len(expression_features) != batch_size:
            raise ValueError("Expression features and identity views must have the same batch size")

        identity_images = torch.cat(identity_images_list, dim=0)
        identity_images = K.geometry.resize(
            identity_images,
            (self.params.image_size, self.params.image_size),
        )
        embeddings, _ = self.encode_identity(identity_images)
        embeddings['ft_expr'] = torch.cat(
            [expression_features] * len(identity_images_list),
            dim=0,
        )

        feature_maps = self.decode(embeddings)['feature_maps']
        return list(feature_maps.split(batch_size, dim=0))

    def forward_identity_only(
            self,
            identity_images_list: list[torch.Tensor],
            cameras_list: list[FoVPerspectiveCameras] | None = None,
            gaussian_renderer: GaussRenderer | None = None,
    ) -> list[dict]:
        """Predict and render one canonical avatar per identity view."""
        if not identity_images_list:
            raise ValueError("Identity-only forward requires at least one image")

        batch_size = len(identity_images_list[0])
        if any(len(images) != batch_size for images in identity_images_list):
            raise ValueError("All identity views must have the same batch size")

        if cameras_list is None:
            cameras_list = [None] * len(identity_images_list)
        if len(cameras_list) != len(identity_images_list):
            raise ValueError("Identity views and cameras must have the same length")

        identity_images = torch.cat([
            K.geometry.resize(images, (self.params.image_size, self.params.image_size))
            for images in identity_images_list
        ], dim=0)
        embeddings, _ = self.encode_identity(identity_images)
        embeddings['ft_expr'] = None
        combined_result = self.decode(embeddings)

        total_batch_size = len(identity_images)
        results = []
        for view_id, cameras in enumerate(cameras_list):
            view_slice = slice(view_id * batch_size, (view_id + 1) * batch_size)
            view_embeddings = {
                name: value[view_slice]
                if torch.is_tensor(value) and value.ndim > 0 and len(value) == total_batch_size
                else value
                for name, value in combined_result['embeddings'].items()
            }
            result = dict(
                embeddings=view_embeddings,
                base_maps=combined_result['base_maps'][view_slice],
                feature_maps=combined_result['feature_maps'][view_slice],
                pos=combined_result['pos'][view_slice],
                scales=combined_result['scales'][view_slice],
                pointclouds=combined_result['pointclouds'][view_slice],
            )
            if cameras is not None:
                if gaussian_renderer is None:
                    raise ValueError("Rendering identity views requires a Gaussian renderer")
                result['pred_images'] = gaussian_renderer.render_images(
                    result['pointclouds'], cameras
                )
            results.append(result)

        results[0]['canonical_maps'] = [result['feature_maps'] for result in results]
        return results

    def forward(
            self,
            x: torch.Tensor,
            x_exp_list: list[torch.Tensor] | None = None,
            cameras_list: list[FoVPerspectiveCameras] | None = None,
            keypoints_list: list[torch.Tensor] | None = None,
            gaussian_renderer: GaussRenderer | None = None,
            is_train=False,
            pred_expr=False,
            identity_images_list: list[torch.Tensor] | None = None,
            identity_only: bool = False,
    ):

        if identity_only:
            if pred_expr:
                raise ValueError("identity_only and pred_expr cannot both be enabled")
            identity_views = x_exp_list if x_exp_list is not None else [x]
            return self.forward_identity_only(
                identity_views,
                cameras_list=cameras_list,
                gaussian_renderer=gaussian_renderer,
            )

        results = []
        x = K.geometry.resize(x, (self.params.image_size, self.params.image_size))
        embeddings, dino_outputs = self.encode_identity(x)
        consistency_expression = None

        if False:
        # if not pred_expr:
            # embeddings['ft_expr'] = None
            embeddings['ft_expr'] = torch.zeros((len(x), 64)).to(self.device)
            outputs = self.decode(embeddings)
            if cameras_list is None:
                res = dict(
                    embeddings=embeddings,
                    base_maps=outputs['base_maps'],
                    feature_maps=outputs['feature_maps'],
                    pos=outputs['pos'],
                    scales=outputs['scales'],
                    pointclouds=outputs['pointclouds'],
                )
                results.append(res)
            else:
                for cameras in cameras_list:
                    res = dict(
                        embeddings=embeddings,
                        base_maps=outputs['base_maps'],
                        feature_maps=outputs['feature_maps'],
                        pos=outputs['pos'],
                        scales=outputs['scales'],
                        pointclouds=outputs['pointclouds'],
                    )
                    res['pred_images'] = gaussian_renderer.render_images(outputs['pointclouds'], cameras)
                    results.append(res)
        else:
            if cameras_list is None:
                cameras_list = [None]

            if x_exp_list is None:
                x_exp_list = [x] * len(cameras_list)

            for x_exp, cameras, keypoints in zip(x_exp_list, cameras_list, keypoints_list):
                if x_exp is not None:
                    x_exp = K.geometry.resize(x_exp, (self.params.image_size, self.params.image_size))
                    dino_outputs = None # do not take dino features from identity images

                if is_train:
                    x_exp = self.expr_augment(x_exp)

                # disp_augs = make_grid(x_exp)
                # show_image("augs", disp_augs, wait=0)

                embeddings['ft_expr'] = self.encode_expressions(x_exp, dino_outputs, keypoints=keypoints)['ft_expr']

                # embeddings['ft_expr'] = self.(keypoints.reshape(-1, 478*3).float())
                embeddings['ft_expr'] = F.normalize(embeddings['ft_expr'])

                if not pred_expr:
                    shuffled_ids = np.random.permutation(range(len(x_exp)))
                    embeddings['ft_expr'] = embeddings['ft_expr'][shuffled_ids]

                if identity_images_list is not None and consistency_expression is None:
                    consistency_expression = embeddings['ft_expr']

                exp_result = self.decode(embeddings)

                if cameras is not None:
                    exp_result['pred_images'] = gaussian_renderer.render_images(exp_result['pointclouds'], cameras)

                results.append(exp_result)

        if identity_images_list is not None:
            if consistency_expression is None:
                raise RuntimeError("Identity consistency requires an expression input")
            alternative_maps = self.predict_canonical_maps(
                identity_images_list,
                consistency_expression,
            )
            results[0]['canonical_maps'] = [results[0]['feature_maps'], *alternative_maps]

        return results


def load_model(net: NeuralHeadNet, filename, snapshot_dir="", return_meta=False) -> NeuralHeadNet | (NeuralHeadNet, dict):
    if os.path.isabs(filename):
        filepath = filename
    else:
        filepath = os.path.join(snapshot_dir, filename)

    if os.path.splitext(filepath)[1] != '.pth':
        filepath = filepath + '.pth'

    snapshot = torch.load(filepath, weights_only=True, map_location="cpu")
    try:
        net.load_state_dict(snapshot['state_dict'], strict=False)
    except RuntimeError as e:
        print(e)

    meta = snapshot['meta']
    str_training_time = str(datetime.timedelta(seconds=meta.get('total_time', 0)))
    log.info("Model {} trained for {} iterations ({}).".format(
        filename, meta['total_iter'], str_training_time )
    )
    if return_meta:
        return net, meta
    else:
        return net
