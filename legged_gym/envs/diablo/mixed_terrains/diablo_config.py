# ==============================================================================
# Author: Zhengkr
# Created on: 2024-03-11
# Description: 双轮足机器人刑天训练配置
# Copyright: Optional, add copyright information if needed.
# Revision History:
#   YYYY-MM-DD: Made modifications, updated XX feature.
# ==============================================================================

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class DiabloRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 151   # 3*6+12+（11*11=36(157)
        num_actions = 6 # 8 → 6 锁死hip关节，并增加hip2关节限位

    
    class terrain( LeggedRobotCfg.terrain):
        # # 暂时在平面上
        # mesh_type = 'plane'
        # measure_heights = False
        
        # 采样机器人周围方形区域内的点
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.3, 0.3, 0., 0.2, 0.2]
        # terrain_proportions = [0., 0., 0., 1., 0.]

        # 跑道测试
        track_test = True
        track_units = ["rough", "wave obstacle", "sloped obstacle", "gap", "pyramid stairs"]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.] # x,y,z [m]
        # todo设置关节初始位置 以及 PD参数
        # 参考传统控制
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # 'hip_left':  0.,
            'hip2_left': 0.,
            'knee_left': 0.,
            'ankle_left': 0.,
            # 'hip_right': 0.,
            'hip2_right': 0.,
            'knee_right': 0.,
            'ankle_right': 0.
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {   'hip': 80., 'hip2': 80.,
                        'knee': 80., 'ankle': 80.     }  # [N*m/rad]
        damping = {   'hip': 5, 'hip2': 5,
                        'knee': 5, 'ankle': 5     }    # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/diablo/urdf/diablo.urdf'
        name = "diablo"
        foot_name = 'wheel'
        terminate_after_contacts_on = ['base']
        flip_visual_attachments = False
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = False   # 设置后无法收敛？4 -1
        base_height_target = 0.5
        class scales(LeggedRobotCfg.rewards.scales):
            termination = 0.             # 惩罚因为非时间耗尽条件而终止的情况
            tracking_lin_vel = 8.0          # 奖励线性速度命令的跟踪（仅x和y轴）
            tracking_ang_vel = 0.5          # 奖励角速度命令的跟踪（仅偏航轴）
            lin_vel_z = -0.5                # 惩罚z轴上的基座线速度
            ang_vel_xy = -0.25               # 惩罚xy轴上的基座角速度
            orientation = -1.5               # 惩罚非水平的基座姿态（xy轴方向上的重力分量）
            torques = -0.00001                # 惩罚力矩的使用
            dof_vel = -0.0                  # 惩罚关节速度
            dof_acc = -0.0                # 惩罚关节加速度
            base_height = -0.              # 惩罚基座高度偏离目标高度
            feet_air_time = 0.              # 奖励长时间的步伐
            collision = -1.                 # 惩罚选定身体部位的碰撞
            action_rate = -0.05            # 惩罚动作的快速变化 | 影响收敛
            feet_stumble = -0.0             # 惩罚脚部撞击垂直表面
            stand_still = -0.               # 在没有运动命令时惩罚机器人的运动
            feet_contact_forces = 0.    # 惩罚足部接触力过大 | 现实是否可以获取
            dof_pos_limits = -1.            # 惩罚关节位置过度运动接近极限
            dof_vel_limits = 0.             # 惩罚关节速度过大接近极限
            torque_limits = 0.              # 惩罚关节力矩过大接近极限
            no_moonwalk = -1.               # 惩罚“太空步”即轮子一前一后
            no_fly = 0.2                   # 惩罚轮子离开地面
            encourage_jump = 0.

    class commands( LeggedRobotCfg.commands ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [0.0, 0.0]     
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

class DiabloRoughCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_diablo'

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01



  