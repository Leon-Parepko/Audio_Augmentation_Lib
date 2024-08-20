import unittest
from Audio_Aug_Lib.Processing.processor import Processor
from constants import *


class TestProcessor(unittest.TestCase):
    def test_processor(self):
        processor = Processor(TEST_PIPELINE_1)
        self.assertEqual(processor.process(MONO_SIGNAL).shape[0], 1)
        processor = Processor(TEST_PIPELINE_2)
        self.assertEqual(processor.process(STEREO_SIGNAL).shape[0], 2)
        processor = Processor(TEST_PIPELINE_3)
        self.assertEqual(processor.process(MONO_SIGNAL).shape[0], 1)
        processor = Processor(TEST_PIPELINE_4)
        self.assertEqual(processor.process(STEREO_SIGNAL).shape[0], 2)

        # Check wrong arguments
        with self.assertRaises(ValueError):
            Processor([])