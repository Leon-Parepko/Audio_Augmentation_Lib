import unittest
from constants import *
from Audio_Aug_Lib.Processing.augmentations import *


class TestAugmentations(unittest.TestCase):
    def test_time_stretch(self):
        # Check the chanel consistency of the output 523264
        self.assertEqual(time_stretch(MONO_SIGNAL).shape[0], 1)
        self.assertEqual(time_stretch(STEREO_SIGNAL).shape[0], 2)

        # Check the length consistency of the output
        self.assertEqual(time_stretch(MONO_SIGNAL, factor=0.5).shape[1], 523264)
        self.assertEqual(time_stretch(MONO_SIGNAL, factor=0.7).shape[1], 733184)
        self.assertEqual(time_stretch(MONO_SIGNAL, factor=1).shape[1], STANDARD_LEN)
        self.assertEqual(time_stretch(MONO_SIGNAL, factor=1.2).shape[1], 1258496)
        self.assertEqual(time_stretch(MONO_SIGNAL, factor=1.5).shape[1], 1572864)

        # Test wrong x shape
        with self.assertRaises(ValueError):
            time_stretch(WRONG_X_1)
        with self.assertRaises(ValueError):
            time_stretch(WRONG_X_2)
        with self.assertRaises(ValueError):
            time_stretch(WRONG_X_3)

        # Check wrong argument values
        with self.assertRaises(ValueError):
            time_stretch(MONO_SIGNAL, factor=0)
        with self.assertRaises(ValueError):
            time_stretch(MONO_SIGNAL, factor=-1)


    def test_pitch_shift(self):
        # Check the chanel consistency of the output
        self.assertEqual(pitch_shift(MONO_SIGNAL).shape[0], 1)
        self.assertEqual(pitch_shift(STEREO_SIGNAL).shape[0], 2)

        # Check the length consistency of the output
        self.assertEqual(pitch_shift(MONO_SIGNAL).shape[1], STANDARD_LEN)
        self.assertEqual(pitch_shift(MONO_SIGNAL, factor=-2).shape[1], STANDARD_LEN)
        self.assertEqual(pitch_shift(MONO_SIGNAL, factor=-1).shape[1], STANDARD_LEN)
        self.assertEqual(pitch_shift(MONO_SIGNAL, factor=0).shape[1], STANDARD_LEN)
        self.assertEqual(pitch_shift(MONO_SIGNAL, factor=1).shape[1], STANDARD_LEN)
        self.assertEqual(pitch_shift(MONO_SIGNAL, factor=2).shape[1], STANDARD_LEN)

        self.assertEqual(pitch_shift(MONO_SIGNAL, factor=0.1, semitones=False).shape[1], STANDARD_LEN)
        self.assertEqual(pitch_shift(MONO_SIGNAL, factor=0.7, semitones=False).shape[1], STANDARD_LEN)
        self.assertEqual(pitch_shift(MONO_SIGNAL, factor=1, semitones=False).shape[1], STANDARD_LEN)
        self.assertEqual(pitch_shift(MONO_SIGNAL, factor=1.2, semitones=False).shape[1], STANDARD_LEN)
        self.assertEqual(pitch_shift(MONO_SIGNAL, factor=1.5, semitones=False).shape[1], STANDARD_LEN)

        # Check wrong x shape
        with self.assertRaises(ValueError):
            pitch_shift(WRONG_X_1)
        with self.assertRaises(ValueError):
            pitch_shift(WRONG_X_2)
        with self.assertRaises(ValueError):
            pitch_shift(WRONG_X_3)

        # Check wrong argument values
        with self.assertRaises(ValueError):
            pitch_shift(MONO_SIGNAL, factor=-0.1, semitones=False)


    def test_noise_addition(self):
        # Check the chanel consistency of the output
        self.assertEqual(noise_addition(MONO_SIGNAL).shape[0], 1)
        self.assertEqual(noise_addition(STEREO_SIGNAL).shape[0], 2)

        # Check the length consistency of the output
        self.assertEqual(noise_addition(MONO_SIGNAL).shape[1], STANDARD_LEN)
        self.assertEqual(noise_addition(MONO_SIGNAL, factor=0.5).shape[1], STANDARD_LEN)
        self.assertEqual(noise_addition(MONO_SIGNAL, factor=1).shape[1], STANDARD_LEN)
        self.assertEqual(noise_addition(MONO_SIGNAL, factor=1.5).shape[1], STANDARD_LEN)
        self.assertEqual(noise_addition(MONO_SIGNAL, factor=2).shape[1], STANDARD_LEN)
        self.assertEqual(noise_addition(MONO_SIGNAL, factor=0.001, noise_type='white').shape[1], STANDARD_LEN)
        self.assertEqual(noise_addition(MONO_SIGNAL, factor=0.001, noise_type='blue').shape[1], STANDARD_LEN)
        self.assertEqual(noise_addition(MONO_SIGNAL, factor=0.001, noise_type='violet').shape[1], STANDARD_LEN)
        self.assertEqual(noise_addition(MONO_SIGNAL, factor=0.001, noise_type='brownian').shape[1], STANDARD_LEN)
        self.assertEqual(noise_addition(MONO_SIGNAL, factor=0.001, noise_type='pink').shape[1], STANDARD_LEN)

        # Check wrong x shape
        with self.assertRaises(ValueError):
            noise_addition(WRONG_X_1, factor=0.001)
        with self.assertRaises(ValueError):
            noise_addition(WRONG_X_2)
        with self.assertRaises(ValueError):
            noise_addition(WRONG_X_3)

        # Check wrong argument values
        with self.assertRaises(ValueError):
            noise_addition(MONO_SIGNAL, factor=0)
        with self.assertRaises(ValueError):
            noise_addition(MONO_SIGNAL, factor=-0.1)
        with self.assertRaises(ValueError):
            noise_addition(MONO_SIGNAL, noise_type='black')


    def test_freq_pass(self):
        # Check the chanel consistency of the output
        self.assertEqual(freq_pass(MONO_SIGNAL, fs=FS, f_type='lowpass').shape[0], 1)
        self.assertEqual(freq_pass(STEREO_SIGNAL, fs=FS, f_type='lowpass').shape[0], 2)
        self.assertEqual(freq_pass(MONO_SIGNAL, fs=FS, f_type='highpass').shape[0], 1)
        self.assertEqual(freq_pass(STEREO_SIGNAL, fs=FS, f_type='highpass').shape[0], 2)

        # Check the length consistency of the output
        self.assertEqual(freq_pass(MONO_SIGNAL, fs=FS, f_type='lowpass').shape[1], STANDARD_LEN)
        self.assertEqual(freq_pass(MONO_SIGNAL, fs=FS, f_type='highpass').shape[1], STANDARD_LEN)
        self.assertEqual(freq_pass(MONO_SIGNAL, fs=2**18, f_type='lowpass').shape[1], STANDARD_LEN)
        self.assertEqual(freq_pass(MONO_SIGNAL, fs=2**18, f_type='highpass').shape[1], STANDARD_LEN)
        self.assertEqual(freq_pass(MONO_SIGNAL, fs=2**15, f_type='lowpass').shape[1], STANDARD_LEN)
        self.assertEqual(freq_pass(MONO_SIGNAL, fs=2**15, f_type='highpass').shape[1], STANDARD_LEN)
        self.assertEqual(freq_pass(MONO_SIGNAL, fs=FS, f_type='lowpass', cutoff=1).shape[1], STANDARD_LEN)
        self.assertEqual(freq_pass(MONO_SIGNAL, fs=FS, f_type='highpass', cutoff=1).shape[1], STANDARD_LEN)
        self.assertEqual(freq_pass(MONO_SIGNAL, fs=FS, f_type='lowpass', cutoff=700).shape[1], STANDARD_LEN)
        self.assertEqual(freq_pass(MONO_SIGNAL, fs=FS, f_type='highpass', cutoff=700).shape[1], STANDARD_LEN)
        self.assertEqual(freq_pass(MONO_SIGNAL, fs=FS, f_type='lowpass', cutoff=5000).shape[1], STANDARD_LEN)
        self.assertEqual(freq_pass(MONO_SIGNAL, fs=FS, f_type='highpass', cutoff=5000).shape[1], STANDARD_LEN)
        self.assertEqual(freq_pass(MONO_SIGNAL, fs=FS, f_type='lowpass', cutoff=10000).shape[1], STANDARD_LEN)
        self.assertEqual(freq_pass(MONO_SIGNAL, fs=FS, f_type='highpass', cutoff=10000).shape[1], STANDARD_LEN)

        # Check wrong x shape
        with self.assertRaises(ValueError):
            freq_pass(WRONG_X_1, fs=FS, f_type='lowpass')
        with self.assertRaises(ValueError):
            freq_pass(WRONG_X_2, fs=FS, f_type='lowpass')
        with self.assertRaises(ValueError):
            freq_pass(WRONG_X_3, fs=FS, f_type='lowpass')

        # Check wrong argument values
        with self.assertRaises(ValueError):
            freq_pass(MONO_SIGNAL, fs=44100, f_type='lowpass', cutoff=0)
        with self.assertRaises(ValueError):
            freq_pass(MONO_SIGNAL, fs=44100, f_type='lowpass', cutoff=-1)
        with self.assertRaises(ValueError):
            freq_pass(MONO_SIGNAL, fs=44100, f_type='miltipass')
        with self.assertRaises(ValueError):
            freq_pass(MONO_SIGNAL, fs=0, f_type='lowpass')
        with self.assertRaises(ValueError):
            freq_pass(MONO_SIGNAL, fs=-1, f_type='lowpass')
        with self.assertRaises(ValueError):
            freq_pass(MONO_SIGNAL, fs=44100, f_type='lowpass', poles=0)
        with self.assertRaises(ValueError):
            freq_pass(MONO_SIGNAL, fs=44100, f_type='lowpass', poles=-1)


    def test_inverse(self):
        # Check the chanel consistency of the output
        self.assertEqual(inverse(MONO_SIGNAL).shape[0], 1)
        self.assertEqual(inverse(STEREO_SIGNAL).shape[0], 2)

        # Check the length consistency of the output
        self.assertEqual(inverse(MONO_SIGNAL).shape[1], STANDARD_LEN)

        # Check wrong x shape
        with self.assertRaises(ValueError):
            inverse(WRONG_X_1)
        with self.assertRaises(ValueError):
            inverse(WRONG_X_2)
        with self.assertRaises(ValueError):
            inverse(WRONG_X_3)


    def test_time_masking(self):
        # Check the chanel consistency of the output
        self.assertEqual(time_masking(MONO_SIGNAL).shape[0], 1)
        self.assertEqual(time_masking(STEREO_SIGNAL).shape[0], 2)

        # Check the length consistency of the output
        self.assertEqual(time_masking(MONO_SIGNAL).shape[1], STANDARD_LEN)
        self.assertEqual(time_masking(MONO_SIGNAL, factor=0.5).shape[1], STANDARD_LEN)
        self.assertEqual(time_masking(MONO_SIGNAL, factor=1).shape[1], STANDARD_LEN)
        self.assertEqual(time_masking(MONO_SIGNAL, factor=1.5).shape[1], STANDARD_LEN)
        self.assertEqual(time_masking(MONO_SIGNAL, factor=2).shape[1], STANDARD_LEN)
        self.assertEqual(time_masking(MONO_SIGNAL, num_masks=0).shape[1], STANDARD_LEN)
        self.assertEqual(time_masking(MONO_SIGNAL, num_masks=1).shape[1], STANDARD_LEN)
        self.assertEqual(time_masking(MONO_SIGNAL, num_masks=5).shape[1], STANDARD_LEN)

        # Check wrong argument values
        with self.assertRaises(ValueError):
            time_masking(MONO_SIGNAL, factor=0)
        with self.assertRaises(ValueError):
            time_masking(MONO_SIGNAL, factor=-0.1)
        with self.assertRaises(ValueError):
            time_masking(MONO_SIGNAL, num_masks=-1)


    def test_frequency_masking(self):
        # Check the chanel consistency of the output
        self.assertEqual(frequency_masking(MONO_SIGNAL, fs=FS).shape[0], 1)
        self.assertEqual(frequency_masking(STEREO_SIGNAL, fs=FS).shape[0], 2)

        # Check the length consistency of the output
        self.assertEqual(frequency_masking(MONO_SIGNAL, fs=FS).shape[1], STANDARD_LEN)
        self.assertEqual(frequency_masking(MONO_SIGNAL, fs=2**18).shape[1], STANDARD_LEN)
        self.assertEqual(frequency_masking(MONO_SIGNAL, fs=2**15).shape[1], STANDARD_LEN)
        self.assertEqual(frequency_masking(MONO_SIGNAL, fs=FS, factor=0.5).shape[1], STANDARD_LEN)
        self.assertEqual(frequency_masking(MONO_SIGNAL, fs=FS, factor=1).shape[1], STANDARD_LEN)
        self.assertEqual(frequency_masking(MONO_SIGNAL, fs=FS, factor=1.5).shape[1], STANDARD_LEN)
        self.assertEqual(frequency_masking(MONO_SIGNAL, fs=FS, factor=2).shape[1], STANDARD_LEN)
        self.assertEqual(frequency_masking(MONO_SIGNAL, fs=FS, num_masks=1).shape[1], STANDARD_LEN)
        self.assertEqual(frequency_masking(MONO_SIGNAL, fs=FS, num_masks=5).shape[1], STANDARD_LEN)

        # Check wrong x shape
        with self.assertRaises(ValueError):
            frequency_masking(WRONG_X_1, fs=FS)
        with self.assertRaises(ValueError):
            frequency_masking(WRONG_X_2, fs=FS)
        with self.assertRaises(ValueError):
            frequency_masking(WRONG_X_3, fs=FS)

        # Check wrong argument values
        with self.assertRaises(ValueError):
            frequency_masking(MONO_SIGNAL, fs=FS, factor=0)
        with self.assertRaises(ValueError):
            frequency_masking(MONO_SIGNAL, fs=FS, factor=-0.1)
        with self.assertRaises(ValueError):
            frequency_masking(MONO_SIGNAL, fs=FS, num_masks=0)
        with self.assertRaises(ValueError):
            frequency_masking(MONO_SIGNAL, fs=FS, num_masks=-1)