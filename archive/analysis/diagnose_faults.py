"""First-episode joint-stratified fault diagnosis in the current play environment.

State/observations are captured before step; termination is captured after step.
Automatic-reset frames are excluded by the alive flag. Does not modify training.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

# Resolve the model used by the original diagnostic script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "quadlocofault_rsl_rl"))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint', required=True)
p.add_argument('--output', required=True)
p.add_argument('--per_joint', type=int, default=2)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--alpha', type=float, default=.1)
p.add_argument('--fault_time', type=float, default=5.)
p.add_argument('--duration', type=float, default=15.2)
p.add_argument('--sample_every', type=int, default=5)
p.add_argument('--save_history', action='store_true')
p.add_argument('--sample_severe', action='store_true', help='Reproduce play uniform [0,alpha] instead of exact alpha.')
p.add_argument('--uniform_material', action='store_true', help='Control run: static/dynamic friction .8 on all shapes.')
p.add_argument('--video', action='store_true')
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.video: a.enable_cameras = True
app = AppLauncher(a).app

import json
import numpy as np
import torch
import gymnasium as gym
from collections import defaultdict
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_quadlocofault_rl.rsl_rl import CustomRslRlVecEnvWrapper
from isaaclab_quadlocofault.mdp.events import randomize_actuator_faults
import isaaclab_quadlocofault_tasks
from models.equiv_gcn_actor import EquivGCNActor
import faulthandler
faulthandler.dump_traceback_later(120, repeat=True)
torch.set_num_threads(4)


def main():
    out = Path(a.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    task = 'EquivGCNMLP-Isaac-Velocity-Rough-Unitree-Go2-Play-v0'
    cfg = parse_env_cfg(task, device=a.device or 'cuda:0', num_envs=13*a.per_joint)
    cfg.seed = a.seed
    cfg.episode_length_s = max(20., a.duration+1)
    terrain = cfg.scene.terrain.terrain_generator
    terrain.seed = a.seed
    selected = terrain.sub_terrains['random_rough']; selected.proportion=1.
    terrain.sub_terrains = {'random_rough': selected}
    # Inject faults explicitly at an exact simulation time, once per first episode.
    cfg.events.randomize_actuator_faults = None
    if a.uniform_material:
        cfg.events.physics_material.params['static_friction_range']=(.8,.8)
        cfg.events.physics_material.params['dynamic_friction_range']=(.8,.8)
    agent = load_cfg_from_registry(task,'rsl_rl_cfg_entry_point')
    agent.seed = a.seed
    dump_yaml(str(out/'env.yaml'), cfg)
    dump_yaml(str(out/'agent.yaml'), agent)
    (out/'protocol.json').write_text(json.dumps(vars(a),indent=2))
    raw = gym.make(task, cfg=cfg, render_mode='rgb_array' if a.video else None)
    if a.video:
        raw = gym.wrappers.RecordVideo(raw, video_folder=str(out/'videos'), step_trigger=lambda s:s==0, video_length=round(a.duration/.02),disable_logger=True)
    env=CustomRslRlVecEnvWrapper(raw,clip_actions=agent.clip_actions)
    base=env.unwrapped; robot=base.scene['robot']; device=base.device
    print('DIAG loading actor',flush=True)
    actor_cfg=agent.actor.to_dict(); actor_cfg.pop('class_name')
    actor=EquivGCNActor(env.get_observations(),12,**actor_cfg).to(device)
    checkpoint=torch.load(str(Path(a.checkpoint).resolve()),map_location=device,weights_only=False)
    actor.load_state_dict(checkpoint['actor_state_dict'],strict=True); actor.eval()
    print('DIAG actor loaded',flush=True)
    n=base.num_envs; dt=base.step_dt
    # Interleave joints to distribute terrain assignments across fault strata.
    assigned=torch.arange(n,device=device)%13
    alive=torch.ones(n,dtype=torch.bool,device=device)
    done_time=torch.full((n,),float('nan'),device=device)
    fault_time=torch.full((n,),float('nan'),device=device)
    causes=np.full(n,'',dtype='<U128')
    foot_names=['FL_foot','FR_foot','RL_foot','RR_foot']
    foot_ids=[robot.body_names.index(x) for x in foot_names]
    sensor=base.scene.sensors['contact_forces']
    sensor_ids=[sensor.body_names.index(x) for x in foot_names]
    records=defaultdict(list)
    def cpu(x):return x.detach().cpu().numpy().copy()
    static=dict(joint_names=np.asarray(robot.joint_names),foot_names=np.asarray(foot_names),assigned_joint=cpu(assigned),defaults_q=cpu(robot.data.default_joint_pos),material_properties=cpu(robot.root_physx_view.get_material_properties()),masses=cpu(robot.root_physx_view.get_masses()),terrain_levels=cpu(base.scene.terrain.terrain_levels),checkpoint=np.asarray(a.checkpoint),horizon_s=np.asarray(a.duration),step_dt_s=np.asarray(dt),sample_dt_s=np.asarray(dt*a.sample_every),scheduled_fault_time_s=np.asarray(a.fault_time),alpha=np.asarray(a.alpha))
    obs=env.get_observations()
    for step in range(round(a.duration/dt)):
        with torch.inference_mode():
            if step==round(a.fault_time/dt):
                ids=torch.nonzero(alive & (assigned<12),as_tuple=False).flatten()
                randomize_actuator_faults(base,ids,SceneEntityCfg('robot'),severe_fault_prob=1. if a.sample_severe else 0.,failure_coef_severe=a.alpha,failure_coef_moderate=a.alpha,num_faults=1,fixed_joint_idx=assigned[ids],apply_once_per_episode=True)
                fault_time[ids]=step*dt
            action=actor(obs)[0]
            if step%a.sample_every==0:
                hits=base.scene.sensors['height_scanner'].data.ray_hits_w[...,2]
                height=robot.data.root_pos_w[:,2]-torch.nanmean(torch.where(torch.isfinite(hits),hits,torch.nan),dim=1)
                logits,_,_=actor.fault_residual_encoder(actor.obs_hist_normalizer(obs['history']))
                frame=dict(time_s=np.asarray(step*dt),alive=alive,fault_mask=robot.faulty_joint_idx,motor_strength=robot.motors_strength,obs_policy=obs['policy'],action=action,q=robot.data.joint_pos,qdot=robot.data.joint_vel,target=robot.data.joint_pos_target,applied_torque=robot.data.applied_torque,computed_torque=robot.data.computed_torque,root_lin_vel_b=robot.data.root_lin_vel_b,root_ang_vel_b=robot.data.root_ang_vel_b,root_quat_w=robot.data.root_quat_w,root_pos_w=robot.data.root_pos_w,base_height=height,foot_force_w=sensor.data.net_forces_w[:,sensor_ids],foot_vel_w=robot.data.body_lin_vel_w[:,foot_ids],foot_pos_w=robot.data.body_pos_w[:,foot_ids],fault_probability=logits.sigmoid())
                if a.save_history:frame['obs_history']=obs['history']
                for key,value in frame.items():records[key].append(cpu(value) if isinstance(value,torch.Tensor) else value)
            obs,_,dones,_=env.step(action)
            first=alive & dones.bool();done_time[first]=(step+1)*dt
            for name in base.termination_manager.active_terms:
                flags=base.termination_manager.get_term(name)
                ids=cpu(torch.nonzero(first & flags,as_tuple=False).flatten())
                for i in ids: causes[i] += ('+' if causes[i] else '')+name
            alive &= ~dones.bool()
            actor.reset(dones)
        if step%100==0:print(f'DIAG step={step} t={step*dt:.2f} first_episode_alive={int(alive.sum())}/{n}',flush=True)
    data={k:np.stack(v) for k,v in records.items()}
    data.update(static,first_done_time_s=cpu(done_time),first_done_cause=causes,actual_fault_time_s=cpu(fault_time))
    np.savez_compressed(out/'rollouts.npz',**data)
    print(f'DIAG saved {out}/rollouts.npz',flush=True)
    faulthandler.cancel_dump_traceback_later()
    env.close()

try:main()
finally:app.close()
