import isaacgym
import torch
from legged_gym.envs.diablo.flat.diablo import DiabloMob
from legged_gym.envs.diablo.flat.diablo_config import DiabloFlatCfg
from legged_gym.utils.math import *
from isaacgym.gymapi import (
    KEY_F,
    KEY_P,
    KEY_L,
    KEY_J,
    KEY_R,
    KEY_U,
    KEY_W,
    KEY_S,
    KEY_A,
    KEY_D,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_UP,
    KEY_DOWN,
    KEY_X,
    KEY_SPACE,
    KEY_M,
    KEY_N,
    KEY_H
)

class KeyboardCtrl:
    def __init__(self, env: DiabloMob, env_cfg: DiabloFlatCfg, **kwargs):
        self.env = env
        self.env_cfg = env_cfg
        self.actions = kwargs["num_actions"]
        self.agent_model = kwargs["agent_model"]
        self.FPV = kwargs["FPV"]
        self.record = False
        
        key_actions = {
            KEY_P: "push_robot",
            KEY_L: "press_robot",
            KEY_J: "action_jitter",
            KEY_R: "agent_full_reset",
            KEY_U: "full_reset",
            KEY_W: "forward",
            KEY_S: "backward",
            KEY_A: "leftturn",
            KEY_D: "rightturn",
            KEY_LEFT: "move_left",
            KEY_RIGHT: "move_right",
            KEY_UP: "move_up",
            KEY_DOWN: "move_down",
            KEY_X: "stop",
            KEY_SPACE: "jump",
            KEY_M: "jump_higher",
            KEY_N: "jump_lower",
            KEY_F: "FPV",
            KEY_H: "record"
        }

        for key, action in key_actions.items():
            env.gym.subscribe_viewer_keyboard_event(env.viewer, key, action)

        # 跳跃高度值
        self.jump_height = torch.full((1,), env_cfg.commands.ranges.jump_height[1])

    def run(self):

        for ui_event in self.env.gym.query_viewer_action_events(self.env.viewer):
            self.env.commands[:, 4] = 0 # 长按没有键盘事件可以连续跳跃
            if ui_event.value == 0:
                continue

            if ui_event.action == "push_robot":
                # manully trigger to push the robot
                self.env._push_robots()
            # if ui_event.action == "press_robot":
            #     env.root_states[:, 9] = torch_rand_float(-env.cfg.domain_rand.max_push_vel_xy, 0, (env.num_envs, 1), device=env.device).squeeze(1)
            #     env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.all_root_states))
            if ui_event.action == "action_jitter":
                # assuming wrong action is taken
                # obs, critic_obs, rews, dones, infos = self.env.step(
                #     torch.tanh(torch.randn_like(self.actions))
                # )
                obs, critic_obs, rews, dones, infos = self.env.step(
                    torch.randn(self.num_actions)
                )
            if ui_event.action == "agent_full_reset":
                self.agent_model.reset()
            if ui_event.action == "full_reset":
                self.agent_model.reset()
                obs, _ = self.env.reset()
            if ui_event.action == "stop":
                self.env.commands[:, [0, 4, 5]] = 0
            if ui_event.action == "forward":
                self.env.commands[:, 0] = torch.clip(self.env.commands[:, 0], 0, 10)
                self.env.commands[:, 0] += 0.5
            if ui_event.action == "backward":
                self.env.commands[:, 0] = torch.clip(self.env.commands[:, 0], -10, 0)
                self.env.commands[:, 0] -= 0.5
            if ui_event.action == "leftturn":
                self.env.commands[:, 3] += 0.5
            if ui_event.action == "rightturn":
                self.env.commands[:, 3] -= 0.5
            if ui_event.action == "move_up":
                self.env.commands[:, 5] += 0.1
            if ui_event.action == "move_down":
                self.env.commands[:, 5] -= 0.1
            if ui_event.action == "jump":
                self.env.commands[:, 5] = 0
                self.env.commands[:, 4] = self.jump_height
            if ui_event.action == "jump_higher":
                self.jump_height += 0.05
            if ui_event.action == "jump_lower":
                self.jump_height -= 0.05
            if ui_event.action == "FPV":
                self.FPV = not self.FPV
            if ui_event.action == "record":
                self.record = not self.record

            self.env.commands[:, 0] = torch.clip(
                self.env.commands[:, 0],
                self.env_cfg.commands.ranges.lin_vel_x[0],
                self.env_cfg.commands.ranges.lin_vel_x[1],
            )
            self.env.commands[:, 2] = torch.clip(
                self.env.commands[:, 2],
                self.env_cfg.commands.ranges.ang_vel_yaw[0],
                self.env_cfg.commands.ranges.ang_vel_yaw[1],
            )
            self.env.commands[:, 3] = wrap_to_pi(self.env.commands[:, 3])

            self.jump_height = torch.clip(
                self.jump_height,
                self.env_cfg.commands.ranges.jump_height[0],
                self.env_cfg.commands.ranges.jump_height[1],
            )

            self.env.commands[:, 5] = torch.clip(
                self.env.commands[:, 5],
                self.env_cfg.commands.ranges.knee_angle[0],
                self.env_cfg.commands.ranges.knee_angle[1],
            )

            print(
                "v:{:.1f} d:{:.2f} h:{:.2f} a:{:.1f}".format(
                    self.env.commands[:, 0].item(),
                    self.env.commands[:, 3].item(),
                    self.jump_height.item(),
                    self.env.commands[:, 5].item(),
                )
            )

            # lo = env_cfg.commands.ranges.heading[0]
            # up = env_cfg.commands.ranges.heading[1]
            # if env.commands[:, 3] > up:
            #     env.commands[:, 3] += lo - up
            # elif env.commands[:, 3] < lo:
            #     env.commands[:, 3] -= lo - up
