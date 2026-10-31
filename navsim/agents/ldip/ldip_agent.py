import os
import pickle
from typing import Any, Union
from typing import Dict
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from tqdm import tqdm

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.gtrs_dense.gtrs_agent import three_to_two_classes
from navsim.agents.gtrs_dense.hydra_features import HydraFeatureBuilder, HydraTargetBuilder
from navsim.agents.ldip.hydra_config import HydraARConfig
from navsim.agents.ldip.hydra_model_ar import HydraARModel
from navsim.common.dataclasses import SensorConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

DEVKIT_ROOT = os.getenv('NAVSIM_DEVKIT_ROOT')
TRAJ_PDM_ROOT = os.getenv('NAVSIM_TRAJPDM_ROOT')


def _nearest_trajectory_indices(
        current_vocab: np.ndarray,
        reference_vocab: np.ndarray,
        chunk_size: int = 512,
) -> torch.Tensor:
    """Map trajectories by squared XY distance over every timepoint."""
    if current_vocab.ndim != 3 or reference_vocab.ndim != 3:
        raise ValueError("Trajectory vocabularies must have shape (N, T, D)")
    if current_vocab.shape[1:] != reference_vocab.shape[1:]:
        raise ValueError(
            "Current and reference vocabularies must have the same trajectory shape, "
            f"got {current_vocab.shape[1:]} and {reference_vocab.shape[1:]}"
        )

    current_xy = torch.from_numpy(
        np.asarray(current_vocab[..., :2])
    ).float().flatten(1)
    reference_xy = torch.from_numpy(
        np.asarray(reference_vocab[..., :2])
    ).float().flatten(1)

    # cdist's Euclidean distance has the same argmin as the summed squared
    # XY distance used by cal_metric_scoring_loss.
    nearest_indices = []
    for start in range(0, current_xy.shape[0], chunk_size):
        distances = torch.cdist(
            current_xy[start:start + chunk_size],
            reference_xy,
            p=2,
            compute_mode='donot_use_mm_for_euclid_dist',
        )
        nearest_indices.append(distances.argmin(dim=1))

    return torch.cat(nearest_indices).long()


def hydra_kd_imi_agent_ar_loss(
        targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: HydraARConfig,
        vocab_pdm_score,
        regression_ep=False,
        three2two=True,
):
    stage_keys = []
    for k in predictions.keys():
        if 'stage' in k:
            stage_keys.append(k)
    stage_keys = sorted(stage_keys)
    loss_dict_overall = {}
    loss_overall = None
    for stage in stage_keys:
        stage_results = predictions[stage]
        loss, loss_dict = ar_loss_per_stage(
            targets, stage_results, config, vocab_pdm_score, regression_ep, three2two
        )
        if loss_overall is None:
            loss_overall = loss
        else:
            loss_overall += loss
        for k, v in loss_dict.items():
            loss_dict_overall[f'{k}_{stage}'] = v

    return loss_overall, loss_dict_overall


def ar_loss_per_stage(
        targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: HydraARConfig,
        vocab_pdm_score_ori,
        regression_ep=False,
        three2two=True,
):
    (no_at_fault_collisions, drivable_area_compliance, time_to_collision_within_bound, ego_progress,
     driving_direction_compliance, lane_keeping, traffic_light_compliance, history_comfort) = (
        predictions['no_at_fault_collisions'],
        predictions['drivable_area_compliance'],
        predictions['time_to_collision_within_bound'],
        predictions['ego_progress'],
        predictions['driving_direction_compliance'],
        predictions['lane_keeping'],
        predictions['traffic_light_compliance'],
        predictions['history_comfort']
    )

    imi = predictions['imi']
    # B L_{t}
    indices = predictions['indices']
    # stage 1
    if vocab_pdm_score_ori['no_at_fault_collisions'].shape[-1] == indices.shape[-1]:
        vocab_pdm_score = vocab_pdm_score_ori
    else:
        vocab_pdm_score = {}
        for k, scores in vocab_pdm_score_ori.items():
            selected_scores = torch.gather(scores, 1, indices)
            vocab_pdm_score[k] = selected_scores
    # 2 cls
    da_loss = F.binary_cross_entropy_with_logits(drivable_area_compliance,
                                                 vocab_pdm_score['drivable_area_compliance'])
    ttc_loss = F.binary_cross_entropy_with_logits(time_to_collision_within_bound,
                                                  vocab_pdm_score['time_to_collision_within_bound'])
    if three2two:
        noc_gt = three_to_two_classes(vocab_pdm_score['no_at_fault_collisions'])
    else:
        noc_gt = vocab_pdm_score['no_at_fault_collisions']
    noc_loss = F.binary_cross_entropy_with_logits(no_at_fault_collisions, noc_gt)

    if regression_ep:
        progress_loss = F.mse_loss(ego_progress.sigmoid(), vocab_pdm_score['ego_progress'])
    else:
        progress_loss = F.binary_cross_entropy_with_logits(ego_progress, vocab_pdm_score['ego_progress'])
    # expansion
    if three2two:
        ddc_gt = three_to_two_classes(vocab_pdm_score['driving_direction_compliance'])
    else:
        ddc_gt = vocab_pdm_score['driving_direction_compliance']
    ddc_loss = F.binary_cross_entropy_with_logits(driving_direction_compliance, ddc_gt)
    lk_loss = F.binary_cross_entropy_with_logits(lane_keeping, vocab_pdm_score['lane_keeping'])
    tl_loss = F.binary_cross_entropy_with_logits(traffic_light_compliance,
                                                 vocab_pdm_score['traffic_light_compliance'])
    comfort_loss = F.binary_cross_entropy_with_logits(history_comfort,
                                                      vocab_pdm_score['history_comfort'])
    vocab = predictions["trajectory_vocab"]
    # B, 8 (4 secs, 0.5Hz), 3
    target_traj = targets["trajectory"]
    # 4, 9, ..., 39
    sampled_timepoints = [5 * k - 1 for k in range(1, 9)]
    l2_distance = -(
            (vocab[:, :, sampled_timepoints] - target_traj[:, None]) ** 2) / config.sigma
    imi_loss = F.cross_entropy(imi, l2_distance.sum((-2, -1)).softmax(1))

    imi_loss_final = config.trajectory_imi_weight * imi_loss
    noc_loss_final = config.trajectory_pdm_weight['no_at_fault_collisions'] * noc_loss
    da_loss_final = config.trajectory_pdm_weight['drivable_area_compliance'] * da_loss
    ttc_loss_final = config.trajectory_pdm_weight['time_to_collision_within_bound'] * ttc_loss
    progress_loss_final = config.trajectory_pdm_weight['ego_progress'] * progress_loss
    ddc_loss_final = config.trajectory_pdm_weight['driving_direction_compliance'] * ddc_loss
    lk_loss_final = config.trajectory_pdm_weight['lane_keeping'] * lk_loss
    tl_loss_final = config.trajectory_pdm_weight['traffic_light_compliance'] * tl_loss
    comfort_loss_final = config.trajectory_pdm_weight['history_comfort'] * comfort_loss

    loss = (
            imi_loss_final
            + noc_loss_final
            + da_loss_final
            + ttc_loss_final
            + progress_loss_final
            + ddc_loss_final
            + lk_loss_final
            + tl_loss_final
            + comfort_loss_final
    )
    loss_dict = {
        'imi_loss': imi_loss_final,
        'pdm_noc_loss': noc_loss_final,
        'pdm_da_loss': da_loss_final,
        'pdm_ttc_loss': ttc_loss_final,
        'pdm_progress_loss': progress_loss_final,
        'pdm_ddc_loss': ddc_loss_final,
        'pdm_lk_loss': lk_loss_final,
        'pdm_tl_loss': tl_loss_final,
        'pdm_comfort_loss': comfort_loss_final
    }

    return loss, loss_dict


class LDiPAgent(AbstractAgent):
    def __init__(
            self,
            config: HydraARConfig,
            lr: float,
            checkpoint_path: str = None,
            pdm_gt_path=None,
            simulator: PDMSimulator = None
    ):
        super().__init__(
            trajectory_sampling=config.trajectory_sampling
        )
        if config.regression_ep:
            ep_lw = 50.0
        else:
            ep_lw = 2.0

        config.trajectory_pdm_weight = {
            'no_at_fault_collisions': 3.0,
            'drivable_area_compliance': 3.0,
            'time_to_collision_within_bound': 4.0,
            'ego_progress': ep_lw,
            'driving_direction_compliance': 1.0,
            'lane_keeping': 2.0,
            'traffic_light_compliance': 3.0,
            'history_comfort': 1.0,
        }
        self._config = config
        self._lr = lr
        self.current_iter = 0
        self.metrics = list(config.trajectory_pdm_weight.keys()) + ['pdm_score']
        self._checkpoint_path = checkpoint_path
        if config.version == 'ar':
            self.model = HydraARModel(config)
        else:
            raise NotImplementedError

        self.vocab_size = config.vocab_size
        self.backbone_wd = config.backbone_wd
        self.scheduler = config.scheduler
        if pdm_gt_path is not None:
            self.vocab_pdm_score_full = pickle.load(
                open(f'{TRAJ_PDM_ROOT}/{pdm_gt_path}', 'rb'))
            tokens = list(self.vocab_pdm_score_full.keys())
            pdm_gt_vocab_size = self.vocab_pdm_score_full[tokens[0]]['no_at_fault_collisions'].shape[0]
            current_vocab = self.model._trajectory_head.vocab.data.cpu().numpy()
            print('pdm_gt_vocab_size:', pdm_gt_vocab_size, 'current_vocab_size:', current_vocab.shape[0])
            self.use_nn_gt = current_vocab.shape[0] < pdm_gt_vocab_size
            if self.use_nn_gt:
                print('Using nearest neighbor for gt')
                largest_vocab_path = os.path.join(
                    os.path.dirname(config.vocab_path),
                    '16384.npy',
                )
                largest_vocab = np.load(largest_vocab_path)
                if largest_vocab.shape[0] != pdm_gt_vocab_size:
                    raise ValueError(
                        f"Reference vocabulary has {largest_vocab.shape[0]} trajectories, "
                        f"but PDM ground truth has {pdm_gt_vocab_size} scores per metric"
                    )

                self.vocab_nn_indices = _nearest_trajectory_indices(
                    current_vocab,
                    largest_vocab,
                )
                score_indices = self.vocab_nn_indices.numpy()

                for token_scores in tqdm(
                        self.vocab_pdm_score_full.values(),
                        total=len(self.vocab_pdm_score_full),
                        desc='Filtering PDM ground-truth scores',
                ):
                    for metric, metric_scores in token_scores.items():
                        if not isinstance(metric_scores, np.ndarray):
                            raise TypeError(
                                f"PDM scores for {metric!r} must be a numpy array, "
                                f"got {type(metric_scores).__name__}"
                            )
                        if metric_scores.shape != (pdm_gt_vocab_size,):
                            raise ValueError(
                                f"PDM scores for {metric!r} must have shape "
                                f"({pdm_gt_vocab_size},), got {metric_scores.shape}"
                            )
                        token_scores[metric] = metric_scores[score_indices]

        if config.ec_target:
            assert simulator is not None
            self.simulator = simulator

    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
        keys_to_delete = [k for k in state_dict if "model._trajectory_head.vocab" in k]
        for k in keys_to_delete:
            del state_dict[k]
        msg = self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()}, strict=False)
        print(msg)

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig(
            cam_f0=[0, 1, 2, 3],
            cam_l0=[0, 1, 2, 3],
            cam_l1=[0, 1, 2, 3],
            cam_l2=[0, 1, 2, 3],
            cam_r0=[0, 1, 2, 3],
            cam_r1=[0, 1, 2, 3],
            cam_r2=[0, 1, 2, 3],
            cam_b0=[0, 1, 2, 3],
            lidar_pc=[],
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [HydraTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [HydraFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(features)

    def evaluate_dp_proposals(self, features: Dict[str, torch.Tensor], dp_proposals) -> Dict[str, torch.Tensor]:
        return self.model.evaluate_dp_proposals(features, dp_proposals)

    def forward_train(self, features, interpolated_traj):
        return self.model(features, interpolated_traj)

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            predictions: Dict[str, torch.Tensor],
            tokens=None,
            prev_ego_state=None
    ):
        # get the pdm score by tokens
        scores = {}
        for k in self.metrics:
            tmp = [self.vocab_pdm_score_full[token][k][None] for token in tokens]
            scores[k] = (
                torch.from_numpy(np.concatenate(tmp, axis=0))
                .to(predictions['trajectory_vocab'].device)
                .to(predictions['trajectory_vocab'].dtype)
            )
        return hydra_kd_imi_agent_ar_loss(
            targets, predictions, self._config, scores,
            regression_ep=self._config.regression_ep,
            three2two=self._config.three2two,
        )

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        backbone_params_name = '_backbone.image_encoder'
        img_backbone_params = list(
            filter(lambda kv: backbone_params_name in kv[0], self.model.named_parameters()))
        default_params = list(filter(lambda kv: backbone_params_name not in kv[0], self.model.named_parameters()))
        params_lr_dict = [
            {'params': [tmp[1] for tmp in default_params]},
            {
                'params': [tmp[1] for tmp in img_backbone_params],
                'lr': self._lr * self._config.lr_mult_backbone,
                'weight_decay': self.backbone_wd
            }
        ]
        if self.scheduler == 'default':
            return torch.optim.Adam(params_lr_dict, lr=self._lr, weight_decay=self._config.weight_decay)
        elif self.scheduler == 'cycle':
            optim = torch.optim.Adam(params_lr_dict, lr=self._lr)
            return {
                "optimizer": optim,
                "lr_scheduler": OneCycleLR(
                    optim,
                    max_lr=0.001,
                    total_steps=20 * 196
                )
            }
        else:
            raise ValueError('Unsupported lr scheduler')

    def get_training_callbacks(self) -> List[pl.Callback]:
        return [
            # TransfuserCallback(self._config),
            ModelCheckpoint(
                save_top_k=100,
                monitor="val/loss_epoch",
                mode="min",
                dirpath=f"{os.environ.get('NAVSIM_EXP_ROOT')}/{self._config.ckpt_path}/",
                filename="{epoch:02d}-{step:04d}",
            )
        ]
