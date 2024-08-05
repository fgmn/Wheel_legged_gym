# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class WoodieFlatCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 30
        num_actions = 6

    
    class terrain( LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.35] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            "lf0_Joint": 0.0,
            "lf1_Joint": 0.0,
            "rf0_Joint": 0.0,
            "rf1_Joint": 0.0,
            "l_wheel_Joint": 0.0,
            "r_wheel_Joint": 0.0,
        }

    class control( LeggedRobotCfg.control ):
        # pos_action_scale = 0.5
        # vel_action_scale = 10.0
        # PD Drive parameters:
        stiffness = {"f0": 80.0, "f1": 80.0, "wheel": 80.}  # [N*m/rad]     # fixed
        damping = {"f0": 5.0, "f1": 5.0, "wheel": 5.0}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/woodie/urdf/robot.urdf'
        name = "woodie"
        penalize_contacts_on = ["lf", "rf", "base"]
        foot_name = 'wheel'
        terminate_after_contacts_on = ["lf", "rf", "base"]
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter 必须设为1
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = False
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -200.
            tracking_lin_vel = 3.0
            tracking_ang_vel = 0.5
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -10.
            torques = -0.00001
            dof_vel = -0.0
            dof_acc = -0.0
            base_height = -0.5
            feet_air_time = 0.
            action_rate = -0.05 
            feet_contact_forces = -0.
            dof_pos_limits = -1.
            no_moonwalk = -0.
            no_fly = 0.2
    
    class commands( LeggedRobotCfg.commands ):
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-2.0, 2.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-1.0, 1.0]   
            heading = [-3.14, 3.14]

class WoodieFlatCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'flat_woodie'

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01



  