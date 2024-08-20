import os.path
import sys
import numpy as np
import scipy.signal as sp
import scipy.io.wavfile
import math
from numba import jit
sys.path.append(os.path.join(os.getcwd()))  # Sorry for this dirty trick :)
from Audio_Aug_Lib.Utils.signal_generators import white_noise, blue_noise, violet_noise, brownian_noise, pink_noise
from Audio_Aug_Lib.Utils.decorators import *




def __interpolate_time(idxs: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """
    Implementation of linear interpolation for time stretching.
    :param idxs: ndarray(num_idxs), the indexes to interpolate.
    """
    start = (idxs + 0.5).astype(int)
    frac = (idxs - start)[None, None, :]
    shifted_arr = np.concatenate((arr[:, :, 1:], np.zeros((arr.shape[0], arr.shape[1], 1))), axis=2)
    return arr[:, :, start] * (1 - frac) + shifted_arr[:, :, start] * frac


@check_argument_pos(['factor'])
@check_x_dim
def time_stretch(x: np.ndarray, factor: float = None) -> np.ndarray:
    """
    Stretch the audio signal in time using phase vocoder method with transient detection (inspired by https://www.youtube.com/watch?v=PjKlMXhxtTM).
    :param x: ndarray(num_channels, num_samples), input signal.
    :param factor: stretching factor. The greater it is, the longer the signal. By default, randomly chosen from recommended range (0.9 to 1.1) (according to https://doi.org/10.21437/Interspeech.2015-711).
    :return: ndarray(num_channels, num_samples * factor), stretched signal.
    """
    # Factor checkups
    if factor is None:
        factor = np.random.uniform(0.9, 1.1)

    # STFT parameters optimized manually
    win_len = 4096
    n_fft = 4096
    hop_len = 1024

    # Perform STFT
    f, t, Zxx = sp.stft(x, nperseg=win_len, noverlap=win_len - hop_len, nfft=n_fft)
    amps = np.abs(Zxx)
    phases = np.angle(Zxx)
    channels, n_freqs, n_frames = Zxx.shape

    # Calculate new frames
    n_new_frames = np.floor(n_frames * factor).astype(int)
    frame_idxs = np.minimum(np.arange(n_new_frames) / factor, n_frames - 1)

    # Calculate phase differences
    phase_diffs = phases - np.concatenate((np.zeros((channels, n_freqs, 1)), phases[:, :, :-1]), axis=2)
    phase_diffs = np.mod(phase_diffs, np.pi * 2)

    # Interpolate and fill the new frames (phases and amplitudes)
    shifted_amps = __interpolate_time(frame_idxs, amps)
    shifted_phase_diffs = __interpolate_time(frame_idxs, phase_diffs)
    unshifted_phases = phases[:, :, (frame_idxs + 0.5).astype(int)]

    # Calculate new phases
    shifted_phases = np.zeros((channels, n_freqs, n_new_frames))
    shifted_phases[:, :, 0] = shifted_phase_diffs[:, :, 0]
    for i in range(1, n_new_frames):
        time_phases = shifted_phases[:, :, i - 1] + shifted_phase_diffs[:, :, i]
        freq_phases = unshifted_phases[:, :, i]

        # Detect transients
        trns_mask = (shifted_amps[:, :, i] - shifted_amps[:, :, i - 1]) / (
                shifted_amps[:, :, i] + shifted_amps[:, :, i - 1])
        trns_mask[trns_mask < 0.5] = 0
        trns_mask[trns_mask >= 0.5] = 1

        # If transient, use unshifted_phases, else use shifted_phases
        shifted_phases[:, :, i] = np.mod(freq_phases * trns_mask + time_phases * (1 - trns_mask),
                                         np.pi * 2)  # mod for readable phases

    # Reconstruct the signal
    Zxx_new = shifted_amps * np.exp(shifted_phases * 1j)
    x_new = sp.istft(Zxx_new, nperseg=win_len, noverlap=win_len - hop_len, nfft=n_fft)[1]
    return x_new


@check_x_dim
def pitch_shift(x: np.ndarray, factor: float = None, semitones: bool = True) -> np.ndarray:
    """
    Change the pitch of the audio signal using resampling of `time_stretch()` result.
    :param x: np.ndarray(num_channels, num_samples), input signal.
    :param factor: By default, it is semitones scaled and randomly chosen from recommended range (-2 to 2. semitones). Can be changed to frequency (hz) scale if needed.
    :param semitones: If True, the factor is treated as semitones. If False - frequency (hz).
    :return: np.ndarray(num_channels, num_samples), pitch-shifted signal.
    """
    if factor is None:
        factor = np.random.uniform(-2, 2)
        semitones = True

    elif factor < 0 and semitones is False:
        raise ValueError("Factor should be positive. Or use semitones=True instead.")

    elif factor == 0:
        return x

    # Convert semitones to hz if needed
    if semitones:
        factor = 2 ** (factor / 12)

    # Stretch and resample
    x_len = x.shape[1]
    x_new = time_stretch(x, factor)
    x_new = sp.resample(x_new, x_len, axis=1)
    return x_new


@check_argument_pos(['factor'])
@check_x_dim
def noise_addition(x: np.ndarray, factor: float = None, noise_type: str = 'white') -> np.ndarray:
    """
    Add colored noise to the signal using generators.
    :param x: ndarray(num_channels, num_samples), input signal.
    :param factor: The gain multiplier for the noise. By default, randomly chosen from recommended range (0.001 to 0.1).
    :param noise_type: 'white' by default. Can be 'blue', 'violet', 'brownian', 'pink'.
    :return: ndarray(num_channels, num_samples), noisy signal.
    """
    if factor is None:
        factor = np.random.uniform(0.001, 0.1)

    # Create noise
    if noise_type == 'white':
        noise = np.array([white_noise(x.shape[1] + 1) for _ in range(x.shape[0])])
    elif noise_type == 'blue':
        noise = np.array([blue_noise(x.shape[1] + 1) for _ in range(x.shape[0])])
    elif noise_type == 'violet':
        noise = np.array([violet_noise(x.shape[1] + 1) for _ in range(x.shape[0])])
    elif noise_type == 'brownian':
        noise = np.array([brownian_noise(x.shape[1] + 1) for _ in range(x.shape[0])])
    elif noise_type == 'pink':
        noise = np.array([pink_noise(x.shape[1]) + 1 for _ in range(x.shape[0])])
    else:
        raise ValueError("Noise type not supported")

    # Ensure the shapes are the same
    if x.shape[1] != noise.shape[1]:
        noise = noise[:, :x.shape[1]]

    x_new = x + factor * noise
    return x_new


@check_argument_pos(arg_names=['fs', 'cutoff', 'poles'])
@check_x_dim
def freq_pass(x: np.ndarray, fs: float, f_type: str, cutoff: float = None, poles: int = 5) -> np.ndarray:
    """
    A fast butterworth filter implementation for high and low pass filtering.
    :param x: ndarray(num_channels, num_samples), input signal.
    :param fs: integer sampling frequency.
    :param f_type: 'lowpass' or 'highpass' for corresponding filtering.
    :param cutoff: The cutoff frequency (hz). By default, randomly chosen from recommended range (100 to 10000 hz).
    :param poles: The number of poles for the butterworth filter. Recommended range is 3 to 10.
    :return: ndarray(num_channels, num_samples), filtered signal.
    """
    if cutoff is None:
        cutoff = np.random.uniform(100, 10000)

    if f_type not in ['lowpass', 'highpass']:
        raise ValueError("Filter type should be 'lowpass' or 'highpass'.")

    # Create a butterworth filter function
    f = sp.butter(poles, cutoff, f_type, fs=fs, output='sos')

    # Apply filter for each channel
    x_new = np.array([scipy.signal.sosfiltfilt(f, x[i, :]) for i in range(x.shape[0])])
    return x_new


@check_x_dim
@jit(nopython=True)
def inverse(x: np.ndarray) -> np.ndarray:
    """
    Simply invert the signal.
    :param x: ndarray(num_channels, num_samples), input signal.
    :return: ndarray(num_channels, num_samples), inverted signal.
    """
    return -x


@jit(nopython=True)
def time_masking(x: np.ndarray, factor: float = None, num_masks: int = None) -> np.ndarray:
    """
    Recursively masks the random time intervals in the signal with zeros.
    :param x: ndarray(num_channels, num_samples), input signal.
    :param factor: The fraction of the masked signal. By default, randomly chosen from recommended range (0.005 to 0.25).
    :param num_masks: Calculates automatically by default. If manual can be any integer.
    :return: ndarray(num_channels, num_samples), masked signal.
    """
    if factor is None:
        factor = np.random.uniform(0.005, 0.25)

    if factor <= 0:
        raise ValueError("Mask factor should be positive.")

    # Calculate the number of masks if not given
    if num_masks is None:
        max_masks = 10
        inv_mask_factor = 1 - factor
        num_masks = math.ceil(inv_mask_factor * (max_masks ** (inv_mask_factor ** 3)))

    channels, x_len = x.shape
    mask_len = int(x_len * factor)

    if num_masks < 0:
        raise ValueError("Number of masks should be positive.")

    if num_masks == 0:
        return x

    else:
        start = np.random.randint(0, x_len - mask_len)
        end = start + mask_len

        x_new = x.copy().T
        x_new[start:end] = 0
        return time_masking(x_new.T, factor, num_masks=(num_masks - 1))


@check_argument_pos(['factor', 'num_masks'])
@check_x_dim
def frequency_masking(x: np.ndarray, fs: float, factor: float = None, num_masks: int = None) -> np.ndarray:
    """
    Masks the random frequency bands using stft spectral transformation.
    :param x: ndarray(num_channels, num_samples), input signal.
    :param fs: Sampling frequency.
    :param factor: The fraction of the signal to be masked. By default, randomly chosen from recommended range (0.1 to 0.7.).
    :param num_masks: Calculates automatically by default. If manual can be any integer.
    :return: ndarray(num_channels, num_samples), masked signal.
    """
    # Factor checkups
    if factor is None:
        factor = np.random.uniform(0.1, 0.7)

    x_new = x.copy()

    # Calculate the number of masks if not given
    if num_masks is None:
        max_masks = 4
        inv_mask_factor = 1 - factor
        num_masks = math.ceil(inv_mask_factor * (max_masks ** inv_mask_factor ** 3))

    # Iterate over each channel
    for i in range(x.shape[0]):

        # Compute the STFT
        f, t, Zxx = sp.stft(x[i, :].T, fs)

        num_freqs = Zxx.shape[0]
        mask_size = math.floor(factor * num_freqs)
        for _ in range(num_masks):
            rand_freq = np.random.randint(0, num_freqs - mask_size)

            # Mask the frequency band
            Zxx[rand_freq:rand_freq + mask_size] = 0

        # Reconstruct the signal
        _, x_new[i, :] = sp.istft(Zxx, fs)

    x_new = np.array(x_new)
    return x_new
