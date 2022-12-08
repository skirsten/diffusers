# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Tuple

import flax
import jax.numpy as jnp

from .scheduling_utils_flax import broadcast_to_shape_from_left


def betas_for_alpha_bar(num_diffusion_timesteps: int, max_beta=0.999) -> jnp.ndarray:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`jnp.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return jnp.array(betas, dtype=jnp.float32)


@flax.struct.dataclass
class SchedulerCommonState:
    alphas: jnp.ndarray
    betas: jnp.ndarray
    alphas_cumprod: jnp.ndarray
    final_alpha_cumprod: jnp.ndarray


def create_common_state(config):
    if config.trained_betas is not None:
        betas = jnp.asarray(config.trained_betas)
    elif config.beta_schedule == "linear":
        betas = jnp.linspace(config.beta_start, config.beta_end, config.num_train_timesteps, dtype=jnp.float32)
    elif config.beta_schedule == "scaled_linear":
        # this schedule is very specific to the latent diffusion model.
        betas = (
            jnp.linspace(
                config.beta_start**0.5, config.beta_end**0.5, config.num_train_timesteps, dtype=jnp.float32
            )
            ** 2
        )
    elif config.beta_schedule == "squaredcos_cap_v2":
        # Glide cosine schedule
        betas = betas_for_alpha_bar(config.num_train_timesteps)
    else:
        # TODO: Does config._class_name exist?
        raise NotImplementedError(f"{config.beta_schedule} does is not implemented for {config._class_name}")

    alphas = 1.0 - betas

    alphas_cumprod = jnp.cumprod(alphas, axis=0)

    # At every step in ddim, we are looking into the previous alphas_cumprod
    # For the final step, there is no previous alphas_cumprod because we are already at 0
    # `set_alpha_to_one` decides whether we set this parameter simply to one or
    # whether we use the final alpha of the "non-previous" one.
    final_alpha_cumprod = jnp.array(1.0) if config.set_alpha_to_one else alphas_cumprod[0]

    return SchedulerCommonState(
        alphas=alphas,
        betas=betas,
        alphas_cumprod=alphas_cumprod,
        final_alpha_cumprod=final_alpha_cumprod,
    )


def add_noise_common(
    state: SchedulerCommonState, original_samples: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray
):
    alphas_cumprod = state.alphas_cumprod

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    sqrt_alpha_prod = broadcast_to_shape_from_left(sqrt_alpha_prod, original_samples.shape)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    sqrt_one_minus_alpha_prod = broadcast_to_shape_from_left(sqrt_one_minus_alpha_prod, original_samples.shape)

    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples
