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

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from .diablo.mixed_terrains.diablo import Diablo
from .diablo.flat.diablo import DiabloMob
from .diablo.flat.diablo_config import DiabloFlatCfg, DiabloFlatCfgPPO
from .diablo.mixed_terrains.diablo_config import DiabloRoughCfg, DiabloRoughCfgPPO
from .woodie.woodie import Woodie
from .woodie.woodie_config import WoodieFlatCfg, WoodieFlatCfgPPO
from .airbot.airbot_config import AirbotRoughCfg, AirbotRoughCfgPPO
from .airbot.airbot_robot import Airbot
from .b2w.b2w_config import B2wRoughCfg, B2wRoughCfgPPO
from .b2w.b2w_robot import B2w
from legged_gym.utils.task_registry import task_registry
import os
# import实例

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )

task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO() )
task_registry.register( "diablo_flat", DiabloMob, DiabloFlatCfg(), DiabloFlatCfgPPO() )
task_registry.register( "diablo_rough", Diablo, DiabloRoughCfg(), DiabloRoughCfgPPO() )
task_registry.register( "Woodie", Woodie, WoodieFlatCfg(), WoodieFlatCfgPPO() )

task_registry.register( "b2w", B2w, B2wRoughCfg(), B2wRoughCfgPPO() )
task_registry.register( "airbot", Airbot, AirbotRoughCfg(), AirbotRoughCfgPPO() )