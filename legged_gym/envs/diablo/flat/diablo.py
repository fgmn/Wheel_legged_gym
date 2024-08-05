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
from legged_gym.utils.math import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
from legged_gym.utils.terrain import Terrain
from .diablo_config import DiabloFlatCfg


class DiabloMob(LeggedRobot):

    def _reward_no_fly(self):  # todo重载奖励函数
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        rew = 1.0 * (torch.sum(1.0 * contacts, dim=1) == 2)
        rew[self.jump] *= -1
        return rew

    def _reward_z_adjust_leg(self):
        # 将膝盖弯曲至指定角度以实现腿长调节
        # theta = torch.stack((self.theta_right, self.theta_left), dim=1)
        # rew = torch.sum(torch.square(self.commands[:, 5:7] - theta), dim=1)
        rew = torch.square(self.commands[:, 5] - self.theta_right)
        rew += torch.square(-self.commands[:, 5] - self.theta_left)  # fixed
        return (self.adjust_leg) * rew

    def _reward_no_moonwalk(self):
        joints = list(self.cfg.init_state.default_joint_angles.keys())
        # hip_right = joints.index('hip_right')
        hip2_right = joints.index("hip2_right")
        knee_right = joints.index("knee_right")
        # hip_left = joints.index('hip_left')
        hip2_left = joints.index("hip2_left")
        knee_left = joints.index("knee_left")
        # 轮子同步时两连杆在x轴方向投影之和相同
        len_ratio = 1.0128  # 连杆长度之比
        # theta_right = self.dof_pos[:, hip_right] + self.dof_pos[:, hip2_right]
        # theta_left = self.dof_pos[:, hip_left] + self.dof_pos[:, hip2_left]
        self.theta_right = self.dof_pos[:, knee_right]
        self.theta_left = self.dof_pos[:, knee_left]
        r = torch.sin(self.theta_right) + len_ratio * torch.sin(
            self.dof_pos[:, hip2_right] + self.theta_right
        )
        l = torch.sin(self.theta_left) + len_ratio * torch.sin(
            self.dof_pos[:, hip2_left] + self.theta_left
        )
        rew = torch.square(r + l)
        return rew

    def _reward_encourage_jump(self):
        # print('base height:', self.root_states[:,2])
        # rew = torch.square(self.root_states[:,2]-0.5)
        # 奖励生效时机器人会进行“跳跃”
        # 奖励高度对时间的积分
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        fly = torch.sum(1.0 * contact, dim=1) == 0  # 处于空中
        first_contact = (self.base_air_time > 0.0) * ~fly
        self.base_air_time += self.dt * torch.clip(
            self.root_states[:, 2],
            torch.tensor(0.0, device=self.device),
            self.commands[:, 4],
        )  # to be fixed

        # rew_airTime = self.base_air_time * first_contact
        rew_airTime = (self.base_air_time - 5e-5) * first_contact
        # rew_airTime = (self.base_air_time - 5e-5) * fly

        # 奖励向上的速度 more critical
        # h = self.commands[:, 4]
        # v = torch.sqrt(-2 * self.gravity_vec[:, 2] * h).to(self.device)
        # now_v = self.root_states[:, 9]
        # rew_airTime = torch.clip(now_v, torch.tensor(0.0, device=self.device), v)

        l, r = self.command_ranges["jump_height"]
        rew_airTime += (
            torch.maximum(torch.tensor(0.0), self.root_states[:, 9])
            * (self.commands[:, 4] - l)
            / (r - l)
        )

        self.base_air_time *= ~fly

        rew_airTime[~self.jump] *= -1
        return rew_airTime

        # 奖励足部发生较大的瞬时接触力

    def _reward_lin_vel_z(self):
        # Penalize z axis linear velocity
        # fix：使用全局速度，否则机器人会通过改变base姿态骗取奖励
        rew = torch.square(self.root_states[:, 9])
        rew[self.jump] *= -1
        return (~self.adjust_leg) * rew

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        # 将当前帧和上一帧的接触情况合并
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt  # 第一次接触地面
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.5) * first_contact, dim=1
        )  # reward only on first contact with the ground
        rew_airTime *= (
            torch.norm(self.commands[:, :2], dim=1) > 0.1
        )  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        rew_airTime[~self.jump] *= -1
        return rew_airTime

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        # return (~self.adjust_leg) * torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation xy轴方向上的重力分量
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        # return (~self.adjust_leg) * torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target 对机体高度有个目标
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )
        # print('base height=', base_height)
        # [-0.0054, -0.0051, -0.0050,  ...,  0.0011, -0.0046, -0.0052] to be fixed
        return (~self.adjust_leg) * torch.square(
            base_height - self.cfg.rewards.base_height_target
        )

    def post_physics_step(self):
        super().post_physics_step()
        # 使用torch.roll滑动窗口更新DoF的历史信息
        self.dof_pos_error_history = torch.roll(
            self.dof_pos_error_history, shifts=-1, dims=1
        )
        self.dof_pos_error_history[:, -1, :] = self.dof_pos - self.default_dof_pos

        self.dof_vel_history = torch.roll(self.dof_vel_history, shifts=-1, dims=1)
        self.dof_vel_history[:, -1, :] = self.dof_vel

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids=env_ids)
        # 清空DoF的历史信息
        self.dof_pos_error_history[env_ids].zero_()
        self.dof_vel_history[env_ids].zero_()

    def _post_physics_step_callback(self):
        env_ids = (
            (
                self.episode_length_buf
                % int(self.cfg.commands.resampling_time / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        # 对命令进行重新采样
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:  # 基于朝向控制进一步推算角速度
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])  # 当前朝向
            # self.commands[:, 2] = torch.clip(
            #     0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0
            # )
            self.commands[:, 2] = torch.clip(
                1.5 * wrap_to_pi(self.commands[:, 3] - heading),
                self.cfg.commands.ranges.ang_vel_yaw[0],
                self.cfg.commands.ranges.ang_vel_yaw[1],
            )

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (
            self.common_step_counter % self.cfg.domain_rand.push_interval == 0
        ):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        # 清除之前命令
        self.commands[env_ids, 4:7] = 0.0

        # 采样跳跃高度命令
        l, r = self.command_ranges["jump_height"]
        jump_heights = torch.rand(len(env_ids), 1, device=self.device) * (r - l) + l
        mask = (
            torch.rand(len(env_ids), 1, device=self.device)
            < self.cfg.commands.threshold
        )
        jump_heights[mask] = 0.0
        self.commands[env_ids, 4] = jump_heights.squeeze(1)
        self.jump = self.commands[:, 4] != 0

        # 采样腿长命令
        l, r = self.command_ranges["knee_angle"]
        knee_angles = torch.rand(len(env_ids), 2, device=self.device) * (r - l) + l

        # for test
        # random_indices = torch.randint(0, 2, (len(env_ids), 2), device=self.device)
        # knee_angles = torch.where(random_indices == 0, torch.tensor(l, device=self.device), torch.tensor(r, device=self.device))

        # 跳跃和腿长调节不能同时执行
        knee_angles[self.jump[env_ids], :] = 0.0

        mask = (
            torch.rand(len(env_ids), 2, device=self.device)
            < self.cfg.commands.threshold
        )
        knee_angles[mask] = 0.0

        self.commands[env_ids, 5:7] = knee_angles
        self.adjust_leg = torch.all(self.commands[:, 5:7] != 0, dim=1)

        if self.cfg.commands.heading_command:  # 基于朝向
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        else:  # 基于角速度
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1) > 0.2
        ).unsqueeze(1)

    def compute_observations(self):
        """Computes observations"""
        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,  # 3
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.projected_gravity,  # 3
                self.commands[:, :3] * self.commands_scale,  # 3
                (self.commands[:, 4] * self.obs_scales.height_measurements).unsqueeze(
                    -1
                ),  # 1
                (self.commands[:, 5:7] * self.obs_scales.dof_pos),  # 2
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 6
                self.dof_vel * self.obs_scales.dof_vel,  # 6
                self.actions,  # 6
            ),
            dim=-1,
        )  # 因此没有地形高度观测值维度为：33

        # 添加关节状态的历史信息后维度为：153
        if self.cfg.env.enable_joint_state_history:
            self.obs_buf = torch.cat(
                (
                    self.obs_buf,
                    self.dof_pos_error_history.view(self.num_envs, -1)
                    * self.obs_scales.dof_pos
                    * self.obs_scales.joint_state_history,
                    self.dof_vel_history.view(self.num_envs, -1)
                    * self.obs_scales.dof_vel
                    * self.obs_scales.joint_state_history,
                ),
                dim=-1,
            )

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:  # 若有地形高度值则加上测量点数:
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # self.obs_scales缩放观测值 归一化
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        x = 3  # 额外命令数
        noise_vec[9 : 12 + x] = 0.0  # commands
        noise_vec[12 + x : 18 + x] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[18 + x : 24 + x] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[24 + x : 30 + x] = 0.0  # previous actions

        if self.cfg.env.enable_joint_state_history:
            noise_vec[30 + x : 90 + x] = (
                noise_scales.dof_pos
                * noise_level
                * self.obs_scales.dof_pos
                * self.obs_scales.joint_state_history
            )
            noise_vec[90 + x : 150 + x] = (
                noise_scales.dof_vel
                * noise_level
                * self.obs_scales.dof_vel
                * self.obs_scales.joint_state_history
            )

        if self.cfg.terrain.measure_heights:  # to be fixed
            noise_vec[48 + x : 235 + x] = (
                noise_scales.height_measurements
                * noise_level
                * self.obs_scales.height_measurements
            )
        return noise_vec

    # TODO: 命令课程
    def update_command_curriculum(self, env_ids):
        """Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        pass
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if (
            torch.mean(self.episode_sums["tracking_lin_vel"][env_ids])
            / self.max_episode_length
            > 0.85 * self.reward_scales["tracking_lin_vel"]
        ):
            # self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            # self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
            # self.cfg.commands.threshold = 0.5
            self.fall_recovery[env_ids] = True  # 学习跌倒恢复
        else:
            self.fall_recovery[env_ids] = False

    def check_termination(self):
        """Check if environments need to be reset"""
        # 条件：（接触力过大视为跌倒 and ~跌倒恢复） or 超时即episode达到最大值
        # self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        reset_condition = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            )
            > 1.0,
            dim=1,
        )
        self.reset_buf = torch.logical_and(
            reset_condition, torch.logical_not(self.fall_recovery)
        )
        self.time_out_buf = (
            self.episode_length_buf > self.max_episode_length
        )  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize actuator network hidden state tensors
        self.base_air_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.jump = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.adjust_leg = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.fall_recovery = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.dof_pos_error_history = torch.zeros(
            self.num_envs,
            self.cfg.env.joint_state_history_length,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.dof_vel_history = torch.zeros(
            self.num_envs,
            self.cfg.env.joint_state_history_length,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

    def _get_env_origins(self):
        """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise create a grid.
        """
        # 设置机器人起始位置
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True  # 自定义起点
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False
            )
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level  # 5
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(
                0, max_init_level + 1, (self.num_envs,), device=self.device
            )
            self.terrain_types = torch.div(
                torch.arange(self.num_envs, device=self.device),
                (self.num_envs / self.cfg.terrain.num_cols),
                rounding_mode="floor",
            ).to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = (
                torch.from_numpy(self.terrain.env_origins)
                .to(self.device)
                .to(torch.float)
            )
            if self.cfg.terrain.track_test:  # 设置起始跑道
                assert (
                    len(self.cfg.terrain.set_origins) == self.num_envs
                ), "Set origins error"
                self.env_origins[:] = self.terrain_origins[
                    torch.tensor(self.cfg.terrain.set_origins).to(self.device),
                    self.terrain_types,
                ]
            else:
                self.env_origins[:] = self.terrain_origins[
                    self.terrain_levels, self.terrain_types
                ]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False
            )
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            # spacing为环境(即机器人)之间的间隔
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0
