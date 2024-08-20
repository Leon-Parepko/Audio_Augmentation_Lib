from tqdm import tqdm
import numpy as np



class Processor:
    def __init__(self, pipeline: list) -> None:
        """
        Initialize the processor with a pipeline.
        :param pipeline: list of functions, the processing pipeline.
        """
        if not pipeline:
            raise ValueError("The pipeline is empty.")
        self.pipeline = pipeline


    def process(self, x: np.ndarray) -> np.ndarray:
        """
        Process the input signal with the pipeline.
        :param x: np.ndarray(num_channels, num_samples), the input signal.
        :return: np.ndarray(num_channels, num_samples), the processed signal. Num samples may change.
        """
        for process in tqdm(self.pipeline, desc="Processing", leave=False, colour="green", ascii="░▒█"):
            x = process(x)
        return x
