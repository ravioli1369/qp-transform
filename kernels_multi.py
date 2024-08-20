"""
Copyright (C) 2024 Riccardo Felicetti <https://github.com/Unoaccaso>

Created Date: Tuesday, January 23rd 2024, 9:16:28 am
Author: Riccardo Felicetti

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the
Free Software Foundation, version 3. This program is distributed in the hope
that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE. See the GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https: //www.gnu.org/licenses/>.
"""

# system libs for package managing
import sys
import os.path

PATH_TO_THIS = os.path.dirname(__file__)
PATH_TO_MASTER = PATH_TO_THIS + "/../../"
sys.path.append(PATH_TO_MASTER)

# cuda section
from cupyx import jit
import cupy
import cupy.typing

@jit.rawkernel()
def compute_Q_fft_phi_tensor(
    phi_values: cupy.typing.NDArray,
    fft_values: cupy.typing.NDArray,
    fft_frequencies: cupy.typing.NDArray,
    Q_values: cupy.typing.NDArray,
    p_value: cupy.float32,
    sampling_rate: cupy.int32,
    fft_phi_plane: cupy.typing.NDArray,
):
    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    phi_idx = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y
    fft_freq_idx = jit.blockIdx.z * jit.blockDim.z + jit.threadIdx.z

    batch_size, num_channels, num_Q, num_phi, num_fft = fft_phi_plane.shape

    if (index < batch_size * num_channels * num_Q):
        batch_idx = index // (num_channels * num_Q)
        channel_idx = (index // num_Q) % num_channels
        Q_idx = index % num_Q

        if (phi_idx < num_phi) and (fft_freq_idx < num_fft):
            phi = phi_values[phi_idx]
            fft_frequency = fft_frequencies[fft_freq_idx]
            fft_value = fft_values[batch_idx, channel_idx, fft_freq_idx]
            Q = Q_values[Q_idx]

            # Wavelet parameters
            q_tilde = Q / cupy.sqrt(1.0 + 2.0j * Q * p_value)
            norm = ((2.0 * cupy.pi / (Q**2)) ** (1.0 / 4.0)) * cupy.sqrt(1.0 / (2.0 * cupy.pi * phi)) * q_tilde
            exponent = ((0.5 * q_tilde * (fft_frequency - phi)) / phi) ** 2

            # The actual wavelet
            wavelet = norm * cupy.exp(-exponent)

            # Scalar product and normalization for fft
            fft_normalization_factor = cupy.sqrt(sampling_rate)
            fft_phi_plane[batch_idx, channel_idx, Q_idx, phi_idx, fft_freq_idx] = fft_value * wavelet * fft_normalization_factor


@jit.rawkernel()
def compute_fft_phi_plane(
    phi_values: cupy.typing.NDArray,
    fft_values: cupy.typing.NDArray,
    fft_frequencies: cupy.typing.NDArray,
    Q_value: cupy.float32,
    p_value: cupy.float32,
    sampling_rate: cupy.int32,
    fft_phi_plane: cupy.typing.NDArray,
):
    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    phi_idx = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y
    fft_freq_idx = jit.blockIdx.z * jit.blockDim.z + jit.threadIdx.z

    batch_size, num_channels, num_phi, num_fft = fft_phi_plane.shape

    if index < batch_size * num_channels:
        batch_idx = index // num_channels
        channel_idx = index % num_channels
        if phi_idx < num_phi and fft_freq_idx < num_fft:
            phi = phi_values[phi_idx]
            fft_frequency = fft_frequencies[fft_freq_idx]
            fft_value = fft_values[batch_idx, channel_idx, fft_freq_idx]

            # Wavelet parameters
            q_tilde = Q_value / cupy.sqrt(1.0 + 2.0j * Q_value * p_value)
            norm = ((2.0 * cupy.pi / (Q_value**2)) ** (1.0 / 4.0)) * cupy.sqrt(1.0 / (2.0 * cupy.pi * phi)) * q_tilde
            exponent = ((0.5 * q_tilde * (fft_frequency - phi)) / phi) ** 2

            # The actual wavelet
            wavelet = norm * cupy.exp(-exponent)

            # Scalar product and normalization for fft
            fft_normalization_factor = cupy.sqrt(sampling_rate)
            fft_phi_plane[batch_idx, channel_idx, phi_idx, fft_freq_idx] = fft_value * wavelet * fft_normalization_factor



@jit.rawkernel()
def compute_energy_density(
    tau_phi_plane: cupy.typing.NDArray,
    kernel_dim: cupy.typing.NDArray,
    out_height: cupy.int32,
    out_width: cupy.int32,
    out_depth: cupy.int32,
    energy_density_GPU: cupy.typing.NDArray,
):
    # Extracting the indeces of the out matrix
    # The input matrix is bigger, it has kernel-dim more element on each border
    Q_idx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    phi_idx = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y
    tau_idx = jit.blockIdx.z * jit.blockDim.z + jit.threadIdx.z

    if (
        (Q_idx < out_height)
        and ((phi_idx >= kernel_dim) and (phi_idx < out_width - kernel_dim - 1))
        and ((tau_idx >= kernel_dim) and (tau_idx < out_depth - kernel_dim - 1))
    ):
        out_phi_idx = phi_idx + kernel_dim
        out_tau_idx = tau_idx + kernel_dim

        # The for cycle is 1 element shorter, because the trapezoidal rule is
        # 1 / 2 * (f(x + 1) - f(x)) * dx
        # dx and dy factors are simplified whe computing the density
        energy_density = 0.0
        for i in range(2 * kernel_dim):
            ker_phi_idx = phi_idx - kernel_dim + i
            for j in range(2 * kernel_dim):
                ker_tau_idx = tau_idx - kernel_dim + i
                ker_tau_idx = out_tau_idx - kernel_dim + j
                energy_11 = (
                    cupy.abs(tau_phi_plane[Q_idx, ker_phi_idx, ker_tau_idx]) ** 2
                )
                energy_12 = (
                    cupy.abs(tau_phi_plane[Q_idx, ker_phi_idx, ker_tau_idx + 1]) ** 2
                )
                energy_21 = (
                    cupy.abs(tau_phi_plane[Q_idx, ker_phi_idx + 1, ker_tau_idx]) ** 2
                )
                energy_density += 1 / 2 * (2 * energy_11 + energy_12 + energy_21)

        energy_density_GPU[Q_idx, out_phi_idx, out_tau_idx] = energy_density
