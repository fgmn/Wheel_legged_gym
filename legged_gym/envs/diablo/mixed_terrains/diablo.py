# ==============================================================================
# Author: Zhengkr
# Created on: 2024-03-11
# Description: 双轮足机器人刑天任务设计
# Copyright: Optional, add copyright information if needed.
# Revision History:
#   YYYY-MM-DD: Made modifications, updated XX feature.
# ==============================================================================


from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float

class Diablo(LeggedRobot):

    def _reward_no_fly(self):   #todo重载奖励函数
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        rew = 1.*(torch.sum(1.*contacts, dim=1)==2)
        return rew

    def _reward_no_moonwalk(self):
        joints = list(self.cfg.init_state.default_joint_angles.keys())
        # hip_right = joints.index('hip_right')
        hip2_right = joints.index('hip2_right')
        knee_right = joints.index('knee_right')
        # hip_left = joints.index('hip_left')
        hip2_left = joints.index('hip2_left')
        knee_left = joints.index('knee_left')
        # 轮子同步时两连杆在x轴方向投影之和相同
        len_ratio = 1.0128  # 连杆长度之比
        # theta_right = self.dof_pos[:, hip_right] + self.dof_pos[:, hip2_right]
        # theta_left = self.dof_pos[:, hip_left] + self.dof_pos[:, hip2_left]
        theta_right = self.dof_pos[:, hip2_right]
        theta_left = self.dof_pos[:, hip2_left]
        r = torch.sin(theta_right) + len_ratio * torch.sin(theta_right + self.dof_pos[:, knee_right])
        l = torch.sin(theta_left) + len_ratio * torch.sin(theta_left + self.dof_pos[:, knee_left])
        rew = torch.square(r + l)
        return rew

    def _reward_encourage_jump(self):
        # print('base height:', self.root_states[:,2])
        # rew = torch.square(self.root_states[:,2]-0.5)
        # 奖励生效时机器人会进行“跳跃”
        # 奖励高度对时间的积分
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        fly = torch.sum(1.*contact, dim=1)==0   # 处于空中
        first_contact = (self.base_air_time > 0.) * ~fly
        self.base_air_time += self.dt * torch.clip(self.root_states[:,2] + 0.05, 0.)

        rew_airTime = (self.base_air_time - 0.05) * first_contact

        # 奖励向上的速度
        rew_airTime += torch.maximum(torch.tensor(0.0), self.base_lin_vel[:, 2])

        self.base_air_time *= ~fly
        return rew_airTime

        # 奖励足部发生较大的瞬时接触力
    
    # def _reward_lin_vel_z(self):
    #     # 奖励向上的速度
    #     return torch.square(torch.maximum(torch.tensor(0.0), self.base_lin_vel[:, 2]))

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # 采样命令——对于台阶地形调整朝向
        pos_x = self.terrain_levels[env_ids].cpu()
        pos_y = self.terrain_types[env_ids].cpu()
        terrain_types = self.terrain.env_type[pos_x, pos_y]
        if isinstance(terrain_types, np.float64):
            terrain_types = np.array([terrain_types])
        # 根据初始化设置取出在台阶地形上的环境
        stairs_ids = np.where((terrain_types >= self.terrain.proportions[1]) & (terrain_types < self.terrain.proportions[3]))
        heading_values = [0., 1.57, -1.57, 3.14, -3.14]
        noise_level = 1e-2
        heading_commands = torch.tensor([heading_values[torch.randint(len(heading_values), (1,)).item()] for _ in stairs_ids], device=self.device)
        heading_commands += torch.randn_like(heading_commands) * noise_level
        self.commands[stairs_ids, 3] = torch.clip(heading_commands, -3.14, 3.14)

        remain_ids = [it for it in env_ids if it not in stairs_ids]
        
        if self.cfg.commands.heading_command:
            self.commands[remain_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(remain_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[remain_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(remain_ids), 1), device=self.device).squeeze(1)

        # if self.cfg.commands.heading_command:   # 基于朝向
        #     self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # else:                                   # 基于角速度
        #     self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        # 3种地形选择：平面、高度场、三角网格
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        # 对命令进行重新采样
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:   # 基于朝向控制进一步推算角速度
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])     # 当前朝向
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
