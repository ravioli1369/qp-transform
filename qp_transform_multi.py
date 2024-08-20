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

# system libs for package managing
import sys
import os.path

from typing import Union

PATH_TO_THIS = os.path.dirname(__file__)
PATH_TO_MASTER = PATH_TO_THIS + "/../../"
sys.path.append(PATH_TO_MASTER)

# cuda
import cupy
import cupy.typing
import cupyx.scipy.fft as cufft

# cpu and typing
import numpy

# locals
from kernels_multi import compute_Q_fft_phi_tensor, compute_fft_phi_plane

BLOCK_SHAPE = (16, 8, 8)


def qp_transform_multi(
    signal_fft_GPU: cupy.typing.NDArray,
    fft_freqs_GPU: cupy.typing.NDArray,
    phi_axis_GPU: cupy.typing.NDArray,
    Q_values_GPU: Union[cupy.typing.NDArray, cupy.float32],
    p_value: numpy.float32,
    sampling_rate: numpy.int32,
) -> cupy.typing.NDArray:
    if isinstance(Q_values_GPU, cupy.ndarray):
        # TODO: voglio usare gli array persistenti (quando serve), per poter calcolare
        # TODO: la qp transform su un dataset di dimensione grande a piacere.

        batch_size, num_channels, num_fft = signal_fft_GPU.shape[:3]

        # preallocating the fft-phi plane.
        Q_fft_phi_tensor = cupy.zeros(
            (
                batch_size,
                num_channels,
                Q_values_GPU.shape[0],
                phi_axis_GPU.shape[0],
                num_fft,
            ),
            dtype=numpy.complex64,
        )
        grid_shape = (
            (batch_size * num_channels * Q_values_GPU.shape[0]) // BLOCK_SHAPE[0]
            + 1,  # X
            phi_axis_GPU.shape[0] // BLOCK_SHAPE[1] + 1,  # Y
            num_fft // BLOCK_SHAPE[2] + 1,  # Z
        )
        height = numpy.int32(Q_fft_phi_tensor.shape[0])
        width = numpy.int32(Q_fft_phi_tensor.shape[1])
        depth = numpy.int32(Q_fft_phi_tensor.shape[2])

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
        normalized_Q_tau_phi_tensor = cufft.ifft(Q_fft_phi_tensor, axis=-1).astype(
            numpy.complex64
        )
        return normalized_Q_tau_phi_tensor

    elif isinstance(Q_values_GPU, numpy.float32):
        # TODO: voglio usare gli array persistenti (quando serve), per poter calcolare
        # TODO: la qp transform su un dataset di dimensione grande a piacere.

        # preallocating the fft-phi plane.
        batch_size, num_channels, num_fft = signal_fft_GPU.shape[:3]
        fft_phi_plane = cupy.zeros(
            (batch_size, num_channels, phi_axis_GPU.shape[0], num_fft),
            dtype=numpy.complex64,
        )
        height = numpy.int32(fft_phi_plane.shape[0])
        width = numpy.int32(fft_phi_plane.shape[1])

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

        normalized_tau_phi_plane = cufft.ifft(fft_phi_plane, axis=-1).astype(
            numpy.complex64
        )
        return normalized_tau_phi_plane

    else:
        raise Exception(
            "Q_values_GPU must be an istance of cupy.ndarray, or numpy.float32"
        )
