from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.dp.dp_model import diff_traj
from navsim.agents.gtrs_dense.hydra_backbone import HydraBackbone
from navsim.agents.ldip.hydra_config import HydraARConfig
from navsim.agents.ldip.tokenization import TrajectoryVocabularyTokenizer
from navsim.agents.utils.attn import MemoryEffTransformer
from navsim.agents.utils.decoder import CrossAttentionDecoderLayer, CrossAttentionDecoder


def generate_block_mask(block_sizes, device='cpu'):
    """
    Args:
        block_sizes: List of ints, e.g., [16384, 2048, 256]
    Returns:
        Boolean mask of shape (L, L) where True is masked (ignored).
    """
    total_L = sum(block_sizes)
    # Initialize with True (mask everything)
    mask = torch.ones((total_L, total_L), dtype=torch.bool, device=device)

    current_row = 0
    for size in block_sizes:
        # For this block of rows, we allow attention to all columns
        # from index 0 up to the end of this current block.
        mask[current_row: current_row + size, : current_row + size] = False
        current_row += size

    return mask


class HydraARModel(nn.Module):
    def __init__(self, config: HydraARConfig):
        super().__init__()

        self._config = config
        self._backbone = HydraBackbone(config)

        img_num = 2 if config.use_back_view else 1
        self._keyval_embedding = nn.Embedding(
            config.img_vert_anchors * config.img_horz_anchors * img_num, config.tf_d_model
        )

        self.downscale_layer = nn.Conv2d(self._backbone.img_feat_c, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear((4 + 2 + 2) * config.num_ego_status, config.tf_d_model)

        self._trajectory_head = HydraTrajHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            nhead=config.vadv2_head_nhead,
            nlayers=config.vadv2_head_nlayers,
            vocab_path=config.vocab_path,
            config=config
        )

        if self._config.ec_target and not self._config.no_cond:
            self.prev_traj_embedding = nn.Sequential(
                nn.Linear(config.trajectory_sampling.num_poses * 3, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, config.tf_d_model),
            )

    def img_feat_blc(self, camera_feature):
        img_features = self._backbone(camera_feature)
        img_features = self.downscale_layer(img_features).flatten(-2, -1)
        img_features = img_features.permute(0, 2, 1)
        return img_features

    @torch.no_grad()
    def prev_frame_inference(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        status_feature: torch.Tensor = features["status_feature"][1]
        camera_feature = features["camera_feature"]
        status_encoding = self._status_encoding(status_feature)
        # original
        if isinstance(camera_feature, list):
            camera_feature = camera_feature[-2]
        img_features = self.img_feat_blc(camera_feature)
        if self._config.use_back_view:
            img_features_back = self.img_feat_blc(features["camera_feature_back"])
            img_features = torch.cat([img_features, img_features_back], 1)
        keyval = img_features
        keyval += self._keyval_embedding.weight[None, ...]
        output: Dict[str, torch.Tensor] = {}
        trajectory = self._trajectory_head(keyval, status_encoding)
        output.update(trajectory)
        return output

    def forward(
            self,
            features: Dict[str, torch.Tensor],
            interpolated_traj=None
    ) -> Dict[str, torch.Tensor]:
        output: Dict[str, torch.Tensor] = {}

        status_feature: torch.Tensor = features["status_feature"][0]
        camera_feature = features["camera_feature"]

        if self._config.num_ego_status == 1 and status_feature.shape[1] == 32:
            status_encoding = self._status_encoding(status_feature[:, :8])
        else:
            status_encoding = self._status_encoding(status_feature)

        # original
        if isinstance(camera_feature, list):
            camera_feature = camera_feature[-1]
        img_features = self.img_feat_blc(camera_feature)
        if self._config.use_back_view:
            img_features_back = self.img_feat_blc(features["camera_feature_back"])
            img_features = torch.cat([img_features, img_features_back], 1)
        keyval = img_features

        keyval += self._keyval_embedding.weight[None, ...]

        if self._config.ec_target:
            prev_inference = self.prev_frame_inference(features)
            output['prev_trajs'] = prev_inference['trajectory']

        trajectory = self._trajectory_head(keyval, status_encoding, interpolated_traj)
        output.update(trajectory)

        return output


class HydraTrajHead(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int, vocab_path: str,
                 nhead: int, nlayers: int, config: HydraARConfig = None
                 ):
        super().__init__()
        assert config.activation in ['relu', 'gelu']

        activation = nn.GELU if config.activation == 'gelu' else nn.ReLU
        activation_func = F.gelu if config.activation == 'gelu' else F.relu
        self.config = config
        self._num_poses = num_poses
        if self.config.cross_only_decoder:
            self.transformer = CrossAttentionDecoder(
                use_checkpoint=config.decoder_checkpoint,
                decoder_layer=CrossAttentionDecoderLayer(
                    d_model,
                    nhead,
                    dim_feedforward=d_ffn,
                    dropout=0.0,
                    batch_first=True,
                ),
                num_layers=nlayers
            )
        else:
            self.transformer = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model, nhead, d_ffn,
                    dropout=0.0, batch_first=True,
                    activation=activation_func
                ), nlayers
            )
        self.vocab = nn.Parameter(
            torch.from_numpy(np.load(vocab_path)),
            requires_grad=False
        )
        if config.extra_head_layers:
            keys = [
                'no_at_fault_collisions',
                'drivable_area_compliance',
                'time_to_collision_within_bound',
                'ego_progress',
                'driving_direction_compliance',
                'lane_keeping',
                'traffic_light_compliance',
                'history_comfort',
                'imi'
            ]
            self.heads = nn.ModuleDict()
            for k in keys:
                self.heads[k] = nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    activation(),
                    nn.Linear(d_ffn, d_ffn),
                    activation(),
                    nn.Linear(d_ffn, 1),
                )
        else:
            self.heads = nn.ModuleDict({
                'no_at_fault_collisions': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    activation(),
                    nn.Linear(d_ffn, 1),
                ),
                'drivable_area_compliance':
                    nn.Sequential(
                        nn.Linear(d_model, d_ffn),
                        activation(),
                        nn.Linear(d_ffn, 1),
                    ),
                'time_to_collision_within_bound': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    activation(),
                    nn.Linear(d_ffn, 1),
                ),
                'ego_progress': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    activation(),
                    nn.Linear(d_ffn, 1),
                ),
                'driving_direction_compliance': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    activation(),
                    nn.Linear(d_ffn, 1),
                ),
                'lane_keeping': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    activation(),
                    nn.Linear(d_ffn, 1),
                ),
                'traffic_light_compliance': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    activation(),
                    nn.Linear(d_ffn, 1),
                ),
                'history_comfort': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    activation(),
                    nn.Linear(d_ffn, 1),
                ),
                'imi': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    activation(),
                    nn.Linear(d_ffn, d_ffn),
                    activation(),
                    nn.Linear(d_ffn, 1),
                )
            })
        length_0 = self.vocab.data.shape[0]
        self.bundle_lengths = []
        self.attn_masks = []
        for i in range(self.config.bundle_num_stages):
            self.bundle_lengths.append(length_0)
            if self.config.bundle_lengths is not None:
                # use hard-coded bundle lengths
                length_0 = self.config.bundle_lengths[i]
            else:
                # use downsample
                length_0 //= self.config.bundle_downsample
        for i in range(self.config.bundle_num_stages):
            if i == 0:
                self.attn_masks.append(None)
            else:
                self.attn_masks.append(generate_block_mask(self.bundle_lengths[:(i + 1)]))

        if self.config.use_fixed_embedding:
            self.traj_token_embedding = nn.Embedding(
                sum(self.bundle_lengths), config.tf_d_model
            )

        # traj tokenization
        self.normalize_vocab_pos = config.normalize_vocab_pos
        if self.normalize_vocab_pos:
            self.encoder = nn.Sequential(
                *[MemoryEffTransformer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=0.0,
                    enc_rms=config.enc_rms
                ) for _ in range(config.encoder_layers)]
            )

        self.use_nerf = config.use_nerf
        if self.config.cnn_traj_tokenization:
            self.pos_embed = TrajectoryVocabularyTokenizer(d_model)
            if self.config.mlp_traj_tokenization:
                self.pos_embed_mlp = nn.Sequential(
                    nn.Linear(num_poses * (3 + 4), d_ffn),
                    activation(),
                    nn.Linear(d_ffn, d_model),
                )
        elif self.config.extra_feat:
            self.pos_embed = nn.Sequential(
                nn.Linear(num_poses * (3 + 4), d_ffn),
                activation(),
                nn.Linear(d_ffn, d_model),
            )
        elif self.use_nerf:
            self.pos_embed = nn.Sequential(
                nn.Linear(1040, d_ffn),
                activation(),
                nn.Linear(d_ffn, d_model),
            )
        else:
            self.pos_embed = nn.Sequential(
                nn.Linear(num_poses * 3, d_ffn),
                activation(),
                nn.Linear(d_ffn, d_model),
            )

    def tokenize_traj(self, vocab, B):
        L = vocab.shape[0]
        if self.config.cnn_traj_tokenization:
            embedded_vocab = self.pos_embed(vocab)[None]
            if self.config.mlp_traj_tokenization:
                extra_vocab = torch.cat([
                    vocab, diff_traj(vocab)
                ], -1)
                embedded_vocab = embedded_vocab + self.pos_embed_mlp(extra_vocab.view(L, -1))[None]
        else:
            extra_vocab = torch.cat([
                vocab, diff_traj(vocab)
            ], -1)
            embedded_vocab = self.pos_embed(extra_vocab.view(L, -1))[None]
        # B 16384 C
        embedded_vocab = self.encoder(embedded_vocab).repeat(B, 1, 1)
        return embedded_vocab

    def forward(self, keyval, status_encoding, interpolated_traj=None) -> Dict[str, torch.Tensor]:
        result = {}
        vocab = self.vocab.data
        device = keyval.device
        L, HORIZON, ACTION_DIM = vocab.shape
        B = keyval.shape[0]
        result["trajectory_vocab"] = self.vocab.data
        embedded_vocab = self.tokenize_traj(vocab, B)
        C = embedded_vocab.shape[-1]
        for stage in range(self.config.bundle_num_stages):
            stage_dict = {}
            # attn_mask = self.attn_masks[stage].to(device) if self.attn_masks[stage] is not None else None
            # stage 1: [16384, 512]
            l = sum(self.bundle_lengths[:(stage + 1)])
            l_prev = sum(self.bundle_lengths[:stage])
            k = self.bundle_lengths[stage]
            pos = self.traj_token_embedding.weight[l_prev:l].unsqueeze(0) if self.config.use_fixed_embedding else 0.0
            if stage == 0:
                current_vocab = self.vocab.data.unsqueeze(0).repeat(B, 1, 1, 1)
                current_embedded_vocab = embedded_vocab
                # absolute indices in vocab 0-16383
                stage_dict['indices'] = torch.arange(0, L, device=device).unsqueeze(0).repeat(B, 1)
            else:
                prev_stage_dict = result[f'stage_results_{stage - 1}']
                # B L_{t-1}
                prev_scores = prev_stage_dict['scores']
                # B L_{t-1}, absolute indices in vocab 0-16383
                prev_indices = prev_stage_dict['indices']
                # B L_{t}, relative indices in 0-L_{t-1}
                if self.training and getattr(self.config, 'use_soft_sampling', True):
                    # Temperature controls the randomness.
                    # High temp = more uniform random, Low temp = approaches hard top-k
                    temperature = getattr(self.config, 'sampling_temperature', 0.1)
                    normalize = getattr(self.config, 'normalize_noise', False)
                    if normalize:
                        normalized_scores = (
                                (prev_scores - prev_scores.mean(dim=-1, keepdim=True)) /
                                (prev_scores.std(dim=-1, keepdim=True).detach() + 1e-6)
                        )

                        # 2. Generate Gumbel noise with high numerical stability via Exponential dist
                        gumbel_noise_norm = torch.randn_like(normalized_scores)

                        # 3. Add noise scaled by temperature
                        noisy_scores = normalized_scores + gumbel_noise_norm * temperature
                    else:
                        # 2. Add noise scaled by temperature to the logits/scores
                        # 1. Generate Gumbel noise
                        uniform_noise = torch.rand_like(prev_scores) + 1e-10
                        gumbel_noise = -torch.log(-torch.log(uniform_noise))
                        noisy_scores = prev_scores + gumbel_noise * temperature

                    # 3. Take top-k of the noisy scores
                    relative_indices = noisy_scores.topk(k, dim=1).indices
                else:
                    # B L_{t}, relative indices in 0-L_{t-1} (Hard Evaluation)
                    relative_indices = prev_scores.topk(k, dim=1).indices

                # B L_{t}, absolute indices in 0-16383
                absolute_indices = torch.gather(prev_indices, 1, relative_indices)
                stage_dict['indices'] = absolute_indices
                # B L_{t} 40 3
                current_vocab = self.vocab.data[absolute_indices]
                # B L_{t} C
                current_embedded_vocab = torch.gather(embedded_vocab, 1, absolute_indices.view(B, k, 1).expand(B, k, C))

            # B L_{t} 40 3  Used in loss calc
            stage_dict["trajectory_vocab"] = current_vocab

            tr_out = self.transformer(current_embedded_vocab + pos, keyval)
            dist_status = tr_out + status_encoding.unsqueeze(1)

            for k, head in self.heads.items():
                stage_dict[k] = head(dist_status).squeeze(-1)

            with torch.no_grad():
                scores = (
                        0.05 * stage_dict['imi'].softmax(-1).log() +
                        0.5 * stage_dict['traffic_light_compliance'].sigmoid().log() +
                        0.5 * stage_dict['no_at_fault_collisions'].sigmoid().log() +
                        0.5 * stage_dict['drivable_area_compliance'].sigmoid().log() +
                        0.5 * stage_dict['driving_direction_compliance'].sigmoid().log() +
                        8.0 * (5 * stage_dict['time_to_collision_within_bound'].sigmoid() +
                               5 * stage_dict['ego_progress'].sigmoid() +
                               5 * stage_dict['lane_keeping'].sigmoid()).log()
                )
                stage_dict['scores'] = scores
            indices_expanded = scores.argmax(1).view(B, 1, 1, 1).expand(-1, 1, HORIZON, ACTION_DIM)
            trajectory = torch.gather(current_vocab, dim=1, index=indices_expanded).squeeze(1)
            # B 40 3
            stage_dict["trajectory"] = trajectory
            result[f'stage_results_{stage}'] = stage_dict

            if stage == (self.config.bundle_num_stages - 1):
                result['trajectory'] = trajectory

        return result
