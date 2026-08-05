from __future__ import annotations

import torch

import isaaclab.utils.string as string_utils
from isaaclab.envs.mdp.actions import actions_cfg, joint_actions
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


class FaultClampJointPositionAction(joint_actions.JointPositionAction):
    """Clamp position targets for every joint belonging to a faulty leg."""

    cfg: "FaultClampJointPositionActionCfg"
    _LEG_PREFIXES = ("FL", "FR", "RL", "RR")

    def __init__(self, cfg: "FaultClampJointPositionActionCfg", env):
        super().__init__(cfg, env)

        constrained_action_ids, _, constraint_limits = string_utils.resolve_matching_names_values(
            cfg.constraint_limits,
            self._joint_names,
            preserve_order=cfg.preserve_order,
        )
        self._constrained_mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
        self._constrained_mask[constrained_action_ids] = True

        self._constraint_lower_limits = torch.full(
            (self.action_dim,),
            -torch.inf,
            dtype=self.processed_actions.dtype,
            device=self.device,
        )
        self._constraint_upper_limits = torch.full(
            (self.action_dim,),
            torch.inf,
            dtype=self.processed_actions.dtype,
            device=self.device,
        )
        limits = torch.as_tensor(
            constraint_limits,
            dtype=self.processed_actions.dtype,
            device=self.device,
        )
        if limits.ndim != 2 or limits.shape[1] != 2:
            raise ValueError(
                "constraint_limits values must be (lower, upper) pairs, "
                f"but received shape {tuple(limits.shape)}."
            )
        if torch.any(limits[:, 0] > limits[:, 1]):
            raise ValueError("Every lower position limit must be less than or equal to its upper limit.")
        self._constraint_lower_limits[constrained_action_ids] = limits[:, 0]
        self._constraint_upper_limits[constrained_action_ids] = limits[:, 1]

        # Map articulation faults to physical legs independently of joint order.
        asset_joint_prefixes = [name[:2] for name in self._asset.joint_names]
        action_joint_prefixes = [name[:2] for name in self._joint_names]
        invalid_prefixes = sorted(
            (set(asset_joint_prefixes) | set(action_joint_prefixes))
            - set(self._LEG_PREFIXES)
        )
        if invalid_prefixes:
            raise ValueError(
                "FaultClampJointPositionAction requires FL/FR/RL/RR joint-name prefixes, "
                f"but found {invalid_prefixes}."
            )

        self._asset_leg_joint_mask = torch.tensor(
            [
                [joint_prefix == leg_prefix for joint_prefix in asset_joint_prefixes]
                for leg_prefix in self._LEG_PREFIXES
            ],
            dtype=torch.bool,
            device=self.device,
        )
        leg_id_by_prefix = {
            prefix: leg_id for leg_id, prefix in enumerate(self._LEG_PREFIXES)
        }
        self._action_leg_ids = torch.tensor(
            [leg_id_by_prefix[prefix] for prefix in action_joint_prefixes],
            dtype=torch.long,
            device=self.device,
        )

    def apply_actions(self):
        position_targets = self.processed_actions.clone()
        if hasattr(self._asset, "faulty_joint_idx"):
            faulty_joints = self._asset.faulty_joint_idx.bool()
            faulty_legs = torch.stack(
                [
                    faulty_joints[:, leg_joint_mask].any(dim=1)
                    for leg_joint_mask in self._asset_leg_joint_mask
                ],
                dim=1,
            )
            fault_mask = faulty_legs.index_select(1, self._action_leg_ids)
            fault_mask &= self._constrained_mask.unsqueeze(0)

            constrained_targets = torch.clamp(
                position_targets,
                min=self._constraint_lower_limits,
                max=self._constraint_upper_limits,
            )
            position_targets = torch.where(fault_mask, constrained_targets, position_targets)
        self._asset.set_joint_position_target(position_targets, joint_ids=self._joint_ids)


@configclass
class FaultClampJointPositionActionCfg(actions_cfg.JointPositionActionCfg):
    """Configuration for faulty-leg position-target clamping."""

    class_type: type[ActionTerm] = FaultClampJointPositionAction

    constraint_limits: dict[str, tuple[float, float]] = {
        ".*_hip_joint": (-0.5, 0.5),
        ".*_thigh_joint": (0.3, 1.3),
        ".*_calf_joint": (-2.1, -0.8),
    }
    """Absolute position-target limits applied to every joint of a faulty leg."""
