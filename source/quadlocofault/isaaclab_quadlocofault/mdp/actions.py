from __future__ import annotations

import torch

import isaaclab.utils.string as string_utils
from isaaclab.envs.mdp.actions import actions_cfg, joint_actions
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


class FaultClampJointPositionAction(joint_actions.JointPositionAction):
    """Joint position action with per-joint target clamping based on fault severity."""

    cfg: "FaultClampJointPositionActionCfg"

    def __init__(self, cfg: "FaultClampJointPositionActionCfg", env):
        super().__init__(cfg, env)

        constrained_action_ids, _ = string_utils.resolve_matching_names(
            cfg.constraint_joint_names, self._joint_names, preserve_order=cfg.preserve_order
        )
        self._constrained_mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
        self._constrained_mask[constrained_action_ids] = True

        self._constraint_lower_limits = torch.full(
            (self.num_envs, self.action_dim), -torch.inf, dtype=self.processed_actions.dtype, device=self.device
        )
        self._constraint_lower_limits[:, constrained_action_ids] = cfg.constraint_lower_limit

    def apply_actions(self):
        position_targets = self.processed_actions.clone()
        if hasattr(self._asset, "motors_strength"):
            fault_mask = self._asset.motors_strength[:, self._joint_ids] < self.cfg.constraint_strength_threshold
            fault_mask &= self._constrained_mask.unsqueeze(0)
            position_targets = torch.where(
                fault_mask,
                torch.maximum(position_targets, self._constraint_lower_limits),
                position_targets,
            )
        self._asset.set_joint_position_target(position_targets, joint_ids=self._joint_ids)


@configclass
class FaultClampJointPositionActionCfg(actions_cfg.JointPositionActionCfg):
    """Configuration for fault-aware joint position target clamping."""

    class_type: type[ActionTerm] = FaultClampJointPositionAction

    constraint_joint_names: str | list[str] = ".*_calf_joint"
    constraint_strength_threshold: float = 0.3
    constraint_lower_limit: float = -2.1
