# ==============================================================================
# Author: Zhengkr
# Created on: 2024-03-11
# Description: 双轮足机器人刑天训练配置
# Copyright: Optional, add copyright information if needed.
# Revision History:
#   YYYY-MM-DD: Made modifications, updated XX feature.
# ==============================================================================

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class DiabloFlatCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 153 # 33
        num_actions = 6
        num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        enable_joint_state_history = True  # 将关节状态历史信息也作为策略网络的输入
        joint_state_history_length = 10

    class terrain(LeggedRobotCfg.terrain):
        add_perlin_noise = True
        # horizontal_scale = 0.025 # [m]
        # comment turn to trimesh
        # mesh_type = "plane"
        measure_heights = False
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain

        # 采样机器人周围方形区域内的点
        measured_points_x = [
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ]  # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5]
        # terrain_proportions = [0., 0., 0., 1., 0.]

        # 跑道测试 for test
        track_test = True
        set_origins = [0]
        track_units = [
            "wave",
            "stone pillars",
            "sloped obstacle",
            "obstacles",
            "pyramid stairs",
        ]


    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.15]  # x,y,z [m]
        # todo设置关节初始位置 以及 PD参数
        # 参考传统控制
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            # 'hip_left':  0.,
            "hip2_left": 0.0,
            "knee_left": 0.0,
            "ankle_left": 0.0,
            # 'hip_right': 0.,
            "hip2_right": 0.0,
            "knee_right": 0.0,
            "ankle_right": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {
            "hip": 80.0,
            "hip2": 80.0,
            "knee": 80.0,
            "ankle": 80.0,
        }  # [N*m/rad]
        damping = {"hip": 5, "hip2": 5, "knee": 5, "ankle": 5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 2

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/diablo/urdf/diablo.urdf"
        name = "diablo"
        foot_name = "wheel"
        penalize_contacts_on = ["base", "leg"]
        terminate_after_contacts_on = ["base", "leg"]  # 防止跪刹
        flip_visual_attachments = False
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1.0, 1.0]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0  # default: 1.

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.0
        only_positive_rewards = True
        base_height_target = 0.5

        class scales(LeggedRobotCfg.rewards.scales):
            termination = 0.0  # 惩罚因为非时间耗尽条件而终止的情况
            tracking_lin_vel = 2.0  # 奖励线性速度命令的跟踪（仅x和y轴）
            tracking_ang_vel = 0.5  # 奖励角速度命令的跟踪（仅偏航轴）
            lin_vel_z = -1.0  # 惩罚z轴上的基座线速度 | 也可以用来奖励跳跃
            ang_vel_xy = -0.25  # 惩罚xy轴上的基座角速度
            orientation = (
                -5.5
            )  # 惩罚非水平的基座姿态（xy轴方向上的重力分量）default: -1.5
            torques = -0.00001  # 惩罚力矩的使用
            dof_vel = -0.0  # 惩罚关节速度
            dof_acc = -0.0  # 惩罚关节加速度
            base_height = -0.0  # 惩罚基座高度偏离目标高度
            feet_air_time = 1.0  # 奖励长时间的步伐
            collision = -180.0  # 惩罚选定身体部位的碰撞
            action_rate = -0.05  # 惩罚动作的快速变化 | 影响收敛
            feet_stumble = -0.0  # 惩罚脚部撞击垂直表面
            stand_still = -0.0  # 在没有运动命令时惩罚机器人的运动
            feet_contact_forces = 0.0  # 惩罚足部接触力过大 | 现实是否可以获取
            dof_pos_limits = -1.0  # 惩罚关节位置过度运动接近极限
            dof_vel_limits = 0.0  # 惩罚关节速度过大接近极限
            torque_limits = 0.0  # 惩罚关节力矩过大接近极限
            no_moonwalk = -2.0  # 惩罚“太空步”即轮子一前一后
            no_fly = 1.0  # 奖励轮子贴合地面
            encourage_jump = 1.5  # 奖励跳跃动作 default：1.0
            z_adjust_leg = -2.0  # 调节腿长

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 2.0
        num_commands = 7  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading, jump，adjust leg(2)
        resampling_time = 20  # time before command are changed[s] default:2
        heading_command = True  # if true: compute ang vel command from heading error
        threshold = 0.5  # 控制技能学习采样频率

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-2.0, 2.0]  # 更高的运动效率
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-5.0, 5.0]
            heading = [-3.14, 3.14]
            jump_height = [0.05, 0.25]  # 跳跃高度
            # knee_angle = [-3.0, -1.0]  # 膝关节弯曲角度
            knee_angle = [-2.3, -1.5]  # 膝关节弯曲角度
    
    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            joint_state_history = 0.25
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        # gravity = [0.0, 0.0, -1.2]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = (
                2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            )


class DiabloFlatCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'gru'
        # rnn_hidden_size = 128
        # rnn_num_layers = 1
        """
        在eth20年的sr中提及用本体感知历史信息可以重建特权信息，提升运动表现
        但是本设计并没有采用特权学习，原因是：
        对于两轮足并不像点足机器人需要通过隐式的方式获取局部地形以进行立足点的规划
        实际测试发现相同设置下添加GRU不如直接MLP
        """

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = "ActorCritic"
        # policy_class_name = 'ActorCriticRecurrent'
        algorithm_class_name = "PPO"
        num_steps_per_env = 24  # per iteration
        max_iterations = 5000  # number of policy updates

        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = "flat_diablo"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
