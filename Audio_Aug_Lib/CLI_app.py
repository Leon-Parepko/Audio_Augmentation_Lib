import os
import argparse
from tqdm import tqdm
from functools import partial
from Utils.tools import read_audio, save_audio
from Processing.augmentations import *
from Processing.processor import Processor
import contextlib


def run():
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Audio file reader and augmenter')
    parser.add_argument('audio_path', type=str, help='Path to the audio file')
    parser.add_argument('-s', '--stereo', action='store_true', help='Convert all audio files to stereo mode (mono by default)')

    args = parser.parse_args()
    audio_path = args.audio_path
    stereo = args.stereo

    # Check the existence of the file or directory
    if not os.path.isfile(audio_path) and not os.path.isdir(audio_path):
        print(f"File '{audio_path}' not found.")
        return

    list_of_files = []
    if os.path.isdir(audio_path):
        for root, dirs, files in os.walk(audio_path):
            for file in files:
                list_of_files.append(os.path.join(root, file))

    elif os.path.isfile(audio_path):
        list_of_files.append(audio_path)

    else:
        print(f"File or directory '{audio_path}' not found.")
        return

    # Iterate over the files
    for audio_path in tqdm(list_of_files, desc="Processing files", unit="file", ascii=" >=", colour="GREEN"):

        # Read the audio file
        try:
            fs, x = read_audio(audio_path, stereo=stereo)
            print(f"File '{audio_path}' loaded successfully.")
        except Exception as e:
            print(f"Error while reading the audio file: {e}")
            return

        # Augment the data
        try:
            pipeline = [
                partial(time_stretch),
                partial(pitch_shift),
                partial(inverse),
                partial(noise_addition),
                partial(time_masking),
                partial(frequency_masking, fs=fs),
                # partial(freq_pass, fs=fs, f_type='lowpass'),  # Frequency masking instead of this
            ]

            processor = Processor(pipeline)

            x_new = processor.process(x)
            print("Augmented successfully.")

        except Exception as e:
            print(f"Error while augmenting the file: {e}")
            return

        # Save the augmented data
        try:
            with contextlib.redirect_stdout(None):
                name, ext = os.path.splitext(audio_path)
                save_audio(x_new, fs, name + "_aug" + ext)
            print("File saved successfully.")
        except Exception as e:
            print(f"Error while saving the audio file: {e}")
            return


if __name__ == "__main__":
    run()