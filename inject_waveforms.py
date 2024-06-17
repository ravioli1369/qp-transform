import random
import numpy as np
import astropy.constants as const

import pycbc.noise, pycbc.psd
from pycbc.waveform import get_td_waveform, taper_timeseries
from pycbc.detector import Detector
from pycbc.types import FrequencySeries
from pycbc.pnutils import nearest_larger_binary_number
from pycbc import DYN_RANGE_FAC
from pycbc.filter import sigma


G = const.G.value
c = const.c.value

def make_padded_frequency_series(vec, filter_N=None, delta_f=None):
        """
        Generate a frequency domain version of the waveform

        Parameters
        ----------
        vec : TimeSeries or FrequencySeries
            The waveform to be transformed
        filter_N : int, optional
            The length of the filter to use in the transformation. If not
            given, a filter length will be chosen automatically.
        delta_f : float, optional
            The frequency spacing of the output FrequencySeries. If not
            given, the frequency spacing of the input TimeSeries will be
            used.

        Returns
        -------
        FrequencySeries
            The frequency domain version of the waveform
        """

        if filter_N is None:
            power = np.ceil(np.log(len(vec), 2)) + 1
            N = 2**power
        else:
            N = filter_N
        n = N // 2 + 1

        curr_length = len(vec)
        new_length = int(nearest_larger_binary_number(curr_length))
        while new_length * vec.delta_t < 1.0 / delta_f:
            new_length = new_length * 2
        vec.resize(new_length)

        v_tilde = vec.to_frequencyseries()
        i_delta_f = v_tilde.get_delta_f()
        v_tilde = v_tilde.numpy()
        df_ratio = int(delta_f / i_delta_f)
        n_freq_len = int((n - 1) * df_ratio + 1)
        assert n <= len(v_tilde)
        df_ratio = int(delta_f / i_delta_f)
        v_tilde = v_tilde[:n_freq_len:df_ratio]
        vectilde = FrequencySeries(v_tilde, delta_f=delta_f, dtype=np.complex64)

        return FrequencySeries(
            vectilde * DYN_RANGE_FAC, delta_f=delta_f, dtype=np.complex128
        )

def get_timeseries(
        m1,
        m2,
        snr,
        eccentricity,
        f_low,
    ):
        """
        Generate strain with a signal injected in it

        Parameters
        ----------
        m1 : float
            The mass of the first body

        m2 : float
            The mass of the second body

        snr : float
            The SNR value to use for the injection

        eccentricity: float
            The Eccentricity value to use for the injection

        f_low : float
            The lower frequency cutoff for the injection

        Returns
        -------
        strain : TimeSeries
            The strain with the injected signal

        hp : TimeSeries
            Waveform signal

        inj : TimeSeries
            Time shifted injected signal

        inj1 : TimeSeries
            Original injected signal
        """

        duration = 32

        flow = f_low
        delta_f = 1.0 / duration
        flen = int(2048 / delta_f) + 1
        psd = pycbc.psd.aLIGOAPlusDesignSensitivityT1800042(flen, delta_f, flow)

        # Generate 32 seconds of noise at 4096 Hz
        seed = random.randint(10, 200)
        random.seed(seed)
        delta_t = 1.0 / 4096
        tsamples = int(duration / delta_t)
        ts = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=seed)

        # inclination runs from 0 to pi, with poles at 0 and pi
        # coa_phase runs from 0 to 2 pi.
        inclination = random.uniform(0, np.pi)
        coa_phase = random.uniform(0, 2 * np.pi)

        declination = random.uniform(0, np.pi)
        right_ascension = random.uniform(0, np.pi)
        polarization = random.uniform(0, np.pi)

        # Injecting signal at fixed location
        end_time = 10  # random.randint(30, 30) in case of ranodm location

        apx = "EccentricTD"
        hp, hc = get_td_waveform(
            approximant=apx,
            mass1=m1,
            mass2=m2,
            eccentricity=eccentricity,
            inclination=inclination,
            coa_phase=coa_phase,
            delta_t=delta_t,
            f_lower=flow,
        )

        det_h1 = Detector("L1")
        signal_h1 = det_h1.project_wave(
            hp, hc, right_ascension, declination, polarization
        )

        signal_h1 = taper_timeseries(signal_h1, "TAPER_STARTEND")
        stilde = make_padded_frequency_series(
            vec=signal_h1, filter_N=flen, delta_f=psd.delta_f
        )
        inj = stilde
        inj /= sigma(stilde, psd=psd, low_frequency_cutoff=flow)
        inj = inj * snr
        inj = inj.to_timeseries(delta_t=ts.delta_t)
        inj1 = inj


        dt = end_time - ts.start_time
        inj = inj.cyclic_time_shift(dt)
        inj.start_time += end_time
        strain = inj + ts
        # strain = strain.whiten(4, 4)

        return strain, hp, inj, inj1