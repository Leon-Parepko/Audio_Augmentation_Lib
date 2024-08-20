import ffmpegio
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.getcwd()))  # Sorry for this dirty trick :)
from Audio_Aug_Lib.Utils.decorators import *



def read_audio(file_path: str, stereo: bool = True) -> tuple:
    """
    Read the audio signal from an audio file.
    :param file_path: the complete path to the file with its name and extension.
    :param stereo: If True, convert to stereo if needed. If False, convert to mono if needed.
    :return: (int, np.ndarray(num_channels, num_samples)) the sampling frequency and the audio signal. """

    # check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'The file or path {file_path} does not exist')

    fs, x = ffmpegio.audio.read(file_path)

    # Convert to stereo if needed
    if stereo and x.shape[1] == 1:
        x = np.hstack((x, x))

    # Convert to mono if needed
    if not stereo and x.shape[1] == 2:
        x = np.mean(x, axis=1)
        x = x[:, np.newaxis]

    # Normalize
    x = x / np.max(np.abs(x))
    return fs, x.T


@check_x_dim
def save_audio(x: np.ndarray, fs: int, file_path: str) -> None:
    """
    Save the audio signal to an audio file.
    :param file_path: the complete path to the file with its name and extension.
    :param fs: integer, the sampling frequency.
    :param x: np.ndarray(num_channels, num_samples), the audio signal.
    :return: None
    """

    # Check if the directory exists
    if not os.path.exists(os.path.dirname(file_path)):
        raise FileNotFoundError(f'The directory {os.path.dirname(file_path)} does not exist')

    # Normalize
    x = x / np.max(np.abs(x))

    # Save
    ffmpegio.audio.write(file_path, fs, x.T, show_log=False)
    return None
