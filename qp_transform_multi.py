"""
Copyright (C) 2024 Riccardo Felicetti <https://github.com/Unoaccaso>

Created Date: Tuesday, January 23rd 2024, 9:20:45 am
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

from typing import Union
import torch
from torchtyping import TensorType

# locals
from kernels_multi import compute_Q_fft_phi_tensor, compute_fft_phi_plane

BLOCK_SHAPE = (16, 8, 8)


def qp_transform_multi(
    signal_fft_GPU: TensorType,
    fft_freqs_GPU: TensorType,
    phi_axis_GPU: TensorType,
    Q_values_GPU: Union[TensorType, float],
    p_value: float,
    sampling_rate: int,
) -> TensorType:
    if isinstance(Q_values_GPU, TensorType):
        # TODO: voglio usare gli array persistenti (quando serve), per poter calcolare
        # TODO: la qp transform su un dataset di dimensione grande a piacere.

        batch_size, num_channels, num_fft = signal_fft_GPU.shape[:3]

        # preallocating the fft-phi plane.
        Q_fft_phi_tensor = torch.zeros(
            (
                batch_size,
                num_channels,
                Q_values_GPU.shape[0],
                phi_axis_GPU.shape[0],
                num_fft,
            ),
            dtype=torch.complex64,
        )
        grid_shape = (
            (batch_size * num_channels * Q_values_GPU.shape[0]) // BLOCK_SHAPE[0]
            + 1,  # X
            phi_axis_GPU.shape[0] // BLOCK_SHAPE[1] + 1,  # Y
            num_fft // BLOCK_SHAPE[2] + 1,  # Z
        )

        # here the qp transform is calculated
        compute_Q_fft_phi_tensor[grid_shape, BLOCK_SHAPE](
            phi_axis_GPU,
            signal_fft_GPU,
            fft_freqs_GPU,
            Q_values_GPU,
            p_value,
            sampling_rate,
            Q_fft_phi_tensor,
        )

        # here the inverse fft is computed and the phi-tau plane is returned
        normalized_Q_tau_phi_tensor = torch.fft.ifft(Q_fft_phi_tensor, axis=-1).astype(
            torch.complex64
        )
        return normalized_Q_tau_phi_tensor

    elif isinstance(Q_values_GPU, float):
        # TODO: voglio usare gli array persistenti (quando serve), per poter calcolare
        # TODO: la qp transform su un dataset di dimensione grande a piacere.

        # preallocating the fft-phi plane.
        batch_size, num_channels, num_fft = signal_fft_GPU.shape[:3]
        fft_phi_plane = torch.zeros(
            (batch_size, num_channels, phi_axis_GPU.shape[0], num_fft),
            dtype=torch.complex64,
        )

        # instatiating cuda variables
        grid_shape = (
            (batch_size * num_channels) // BLOCK_SHAPE[0] + 1,  # X
            phi_axis_GPU.shape[0] // BLOCK_SHAPE[1] + 1,  # Y
            num_fft // BLOCK_SHAPE[2] + 1,  # Z
        )

        compute_fft_phi_plane[grid_shape, BLOCK_SHAPE](
            phi_axis_GPU,
            signal_fft_GPU,
            fft_freqs_GPU,
            Q_values_GPU,
            p_value,
            sampling_rate,
            fft_phi_plane,
        )

        normalized_tau_phi_plane = torch.fft.ifft(fft_phi_plane, axis=-1).astype(
            torch.complex64
        )
        return normalized_tau_phi_plane

    else:
        raise Exception("Q_values_GPU must be an instance of TensorType, or float")
