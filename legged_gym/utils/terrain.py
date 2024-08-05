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

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import matplotlib.pyplot as plt


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", "plane"]:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        # 求前缀和得到每种地形的choice区间
        self.proportions = [
            np.sum(cfg.terrain_proportions[: i + 1])
            for i in range(len(cfg.terrain_proportions))
        ]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        # 记录地形类型，用于对台阶地形调整命令
        self.env_type = np.zeros((cfg.num_rows, cfg.num_cols))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)
        # 边界环绕整个地形
        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        elif cfg.track_test:
            self.build_track()  # only for test
        else:
            self.randomized_terrain() 

        # 添加柏林噪声
        if cfg.add_perlin_noise:
            self.xSize = cfg.terrain_length * cfg.num_rows + 2 * cfg.border_size
            self.ySize = cfg.terrain_width * cfg.num_cols + 2 * cfg.border_size
            # print('Perlin:', self.xSize, self.ySize, self.tot_rows, self.tot_cols)
            self.perlin_noise = self.generate_fractal_noise_2d(
                self.xSize, self.ySize, self.tot_rows, self.tot_cols
            )
            # plt.imshow(self.perlin_noise, cmap="gray")
            # plt.colorbar()
            # plt.show()
            # np.savetxt("perlin_noise.txt", self.perlin_noise, fmt="%f", delimiter=",")
            self.height_field_raw += (self.perlin_noise / cfg.vertical_scale).astype(
                np.int16
            )

        self.heightsamples = self.height_field_raw
        # np.savetxt('heightsamples.txt', self.heightsamples, fmt='%d', delimiter=',')

        if self.type == "trimesh":
            self.vertices, self.triangles = (
                terrain_utils.convert_heightfield_to_trimesh(
                    self.height_field_raw,
                    self.cfg.horizontal_scale,
                    self.cfg.vertical_scale,
                    self.cfg.slope_treshold,
                )
            )

    @staticmethod
    def generate_perlin_noise_2d(shape, res):
        def f(t):
            return 6 * t**5 - 15 * t**4 + 10 * t**3

        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        # print('log:', delta, d)
        grid = (
            np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0)
            % 1
        )
        # Gradients
        angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
        g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
        # Ramps
        n00 = np.sum(grid * g00, 2)
        n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
        n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
        # Interpolation
        t = f(grid)
        n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
        n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
        return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1) * 0.5 + 0.5

    @staticmethod
    def generate_fractal_noise_2d(
        xSize=20,
        ySize=20,
        xSamples=1600,
        ySamples=1600,
        frequency=1,
        fractalOctaves=2,
        fractalLacunarity=2.0,
        fractalGain=0.25,
        zScale=0.2,
    ):
        xScale = int(frequency * xSize)
        yScale = int(frequency * ySize)
        amplitude = 1
        shape = (xSamples, ySamples)
        noise = np.zeros(shape)
        for _ in range(fractalOctaves):  # 迭代生成多层级的Perlin噪声
            noise += (
                amplitude
                * Terrain.generate_perlin_noise_2d(
                    (xSamples, ySamples), (xScale, yScale)
                )
                * zScale
            )
            amplitude *= fractalGain
            xScale, yScale = int(fractalLacunarity * xScale), int(
                fractalLacunarity * yScale
            )

        return noise

    def build_track(self):
        # 搭建测试跑道
        assert self.cfg.num_rows >= len(
            self.cfg.track_units
        ), "Rows isn't enough for track."
        for x, unit in enumerate(self.cfg.track_units):
            terrain = self.add_terrain(unit)
            self.add_terrain_to_map(terrain, x, 0)

    def add_terrain(self, unit):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        # 通过字典+lambda函数构建选择器
        # 选择跑道单元
        f = {
            "rough": lambda: terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-0.05,
                max_height=0.05,
                step=0.005,
                downsampled_scale=0.2,
            ),
            "sloped": lambda: terrain_utils.sloped_terrain(terrain, slope=-0.5),
            "pyramid sloped": lambda: terrain_utils.pyramid_sloped_terrain(
                terrain, slope=0.2, platform_size=3.0
            ),
            "sloped obstacle": lambda: sloped_obstacle(terrain, slope=0.3),
            "stairs": lambda: terrain_utils.stairs_terrain(
                terrain, step_width=0.75, step_height=-0.5
            ),
            "pyramid stairs": lambda: terrain_utils.pyramid_stairs_terrain(
                terrain, step_width=0.31, step_height=-0.1, platform_size=3.0
            ),
            "obstacles": lambda: terrain_utils.discrete_obstacles_terrain(
                terrain,
                max_height=0.2,
                min_size=1.0,
                max_size=2.0,
                num_rects=15,
                platform_size=3.0,
            ),
            "wave": lambda: terrain_utils.wave_terrain(
                terrain, num_waves=5.0, amplitude=0.15
            ),
            "gap": lambda: gap_terrain(
                terrain,
                size=0.5,
            ),
            "stone pillars": lambda: stone_pillars_terrain(
                terrain, num_pillars=5, pillar_height=0.5, pillar_diameter=0.2
            ),
        }.get(unit)

        f()
        return terrain

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.env_type[i, j] = choice  # ∈ [0,1]
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.env_type[i, j] = choice
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop("type")
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.width_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale,
            )

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.env_type[i, j] = choice
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        # f(difficulty)生成地形的难度等级
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1.0 * difficulty
        pit_depth = 1.0 * difficulty
        amp = 0.15 * difficulty

        if choice < self.proportions[0]:  # 金字塔斜坡
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=slope, platform_size=3.0
            )
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=slope, platform_size=3.0
            )
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-0.05,
                max_height=0.05,
                step=0.005,
                downsampled_scale=0.2,
            )
        elif choice < self.proportions[3]:  # 台阶
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(
                terrain, step_width=0.31, step_height=step_height, platform_size=3.0
            )
        elif choice < self.proportions[4]:  # 离散障碍物
            num_rectangles = 20
            rectangle_min_size = 1.0
            rectangle_max_size = 2.0
            terrain_utils.discrete_obstacles_terrain(
                terrain,
                discrete_obstacles_height,
                rectangle_min_size,
                rectangle_max_size,
                num_rectangles,
                platform_size=3.0,
            )
        elif choice < self.proportions[5]:  # 跳石地形
            terrain_utils.stepping_stones_terrain(
                terrain,
                stone_size=stepping_stones_size,
                stone_distance=stone_distance,
                max_height=0.0,
                platform_size=4.0,
            )
        elif choice < self.proportions[6]:  # 间隙地形
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.0)
        else:  # 坑洞地形 or 波浪地形
            # pit_terrain(terrain, depth=pit_depth, platform_size=4.)
            terrain_utils.wave_terrain(terrain, num_waves=5.0, amplitude=0.1)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        # 在2×2的方形区域内采样高度最大值作为原点z值
        x1 = int((self.env_length / 2.0 - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2.0 + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2.0 - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2.0 + 1) / terrain.horizontal_scale)
        env_origin_z = (
            np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        )
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


def gap_terrain(terrain, gap_size, platform_size=1.0):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[
        center_x - x2 : center_x + x2, center_y - y2 : center_y + y2
    ] = -1000
    terrain.height_field_raw[
        center_x - x1 : center_x + x1, center_y - y1 : center_y + y1
    ] = 0


def pit_terrain(terrain, depth, platform_size=1.0):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth


def sloped_obstacle(terrain, slope=0.3):
    # 构建飞坡障碍
    start = 3 * terrain.width // 8
    end = 5 * terrain.width // 8
    x = np.arange(start, end)
    y = np.arange(start, end)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(len(x), 1)
    xx -= xx[0, 0]
    max_height = int(
        slope * (terrain.horizontal_scale / terrain.vertical_scale) * terrain.width
    )
    terrain.height_field_raw[start:end, start:end] += (
        max_height * xx / terrain.width
    ).astype(terrain.height_field_raw.dtype)
    return terrain


def stone_pillars_terrain(
    terrain, num_pillars=5, pillar_height=0.5, pillar_diameter=1.0
):
    # 创建石柱阵列地形
    pillar_height = int(pillar_height / terrain.vertical_scale)
    pillar_diameter = int(pillar_diameter / terrain.horizontal_scale / 2)

    spacing_x = terrain.length // (num_pillars + 1)
    spacing_y = terrain.width // (num_pillars + 1)

    for i in range(1, num_pillars + 1):
        for j in range(1, num_pillars + 1):
            center_x = i * spacing_x
            center_y = j * spacing_y

            x1 = max(0, center_x - pillar_diameter)
            x2 = min(terrain.length, center_x + pillar_diameter)
            y1 = max(0, center_y - pillar_diameter)
            y2 = min(terrain.width, center_y + pillar_diameter)

            terrain.height_field_raw[x1:x2, y1:y2] = pillar_height

    return terrain
