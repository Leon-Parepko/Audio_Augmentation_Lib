# ---- This file contains constants are used for unit testing purposes. ---- #


from functools import partial
from Audio_Aug_Lib.Processing.augmentations import *
import numpy as np

STANDARD_LEN = 1024 ** 2
MONO_SIGNAL = np.random.randn(1, STANDARD_LEN)
STEREO_SIGNAL = np.random.randn(2, STANDARD_LEN)
WRONG_X_1 = np.random.rand(3, 2085)
WRONG_X_2 = np.random.rand(29, 12, 33)
WRONG_X_3 = np.random.rand(10)
FS = 44100
TEST_AUDIO_PATH = 'test_audio/unit_tests.wav'
TEST_PIPELINE_1 = [
    partial(pitch_shift, factor=-1.5, semitones=True),
    partial(time_stretch, factor=1.5),
    partial(noise_addition, factor=0.1, noise_type="violet"),
    partial(freq_pass, fs=FS, f_type='lowpass', cutoff=3000, poles=5),
    partial(inverse),
    partial(time_masking, factor=0.05, num_masks=3),
    partial(frequency_masking, fs=FS, factor=0.1, num_masks=3)
]
TEST_PIPELINE_2 = [
    partial(pitch_shift),
    partial(time_stretch),
    partial(noise_addition),
    partial(freq_pass, fs=FS, f_type='highpass', cutoff=3000),
    partial(inverse),
    partial(time_masking),
    partial(frequency_masking, fs=FS)
]
TEST_PIPELINE_3 = [
    partial(freq_pass, fs=FS, f_type='highpass', cutoff=3000),
    partial(pitch_shift),
    partial(inverse),
    partial(frequency_masking, fs=FS),
    partial(noise_addition),
    partial(pitch_shift),
    partial(frequency_masking, fs=FS),
]
TEST_PIPELINE_4 = [
    partial(pitch_shift),
    partial(pitch_shift),
    partial(pitch_shift),
]