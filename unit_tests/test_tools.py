import unittest
from Audio_Aug_Lib.Utils.tools import *
from constants import *



class TestTools(unittest.TestCase):
    def test_read_audio(self):
        # Check shape consistency
        fs, x = read_audio(TEST_AUDIO_PATH)
        self.assertEqual(x.shape, (2, 208512))
        self.assertEqual(fs, 48000)
        fs, x = read_audio(TEST_AUDIO_PATH, stereo=True)
        self.assertEqual(x.shape, (2, 208512))
        self.assertEqual(fs, 48000)
        fs, x = read_audio(TEST_AUDIO_PATH, stereo=False)
        self.assertEqual(x.shape, (1, 208512))
        self.assertEqual(fs, 48000)

        # Check wrong arguments
        with self.assertRaises(FileNotFoundError):
            read_audio('unit_tests_wrong.wav')
        with self.assertRaises(FileNotFoundError):
            read_audio('')


    def test_save_audio(self):
        # Check wrong arguments
        with self.assertRaises(ValueError):
            save_audio(WRONG_X_1, 48000, 'test_audio/unit_tests.wav')
        with self.assertRaises(ValueError):
            save_audio(WRONG_X_2, 48000, 'test_audio/unit_tests.wav')
        with self.assertRaises(ValueError):
            save_audio(WRONG_X_3, 48000, 'test_audio/unit_tests.wav', aaa=-23)
        with self.assertRaises(FileNotFoundError):
            save_audio(MONO_SIGNAL, 48000, 'wrong_directory/unit_tests.wav')


