# Audio Augmentation For Speech Recognition

This is a simple audio augmentation package that contains algorithms implemented using only `scipy` and `numpy`. The package consists of the `processor.py` module that can be used directly in your code and of the CLI application (`CLI_app.py`) that can be run from the terminal.

The program works with `.wav` files because of raw data representation. The processing of `mp3`, `flac`, etc. files is supported, but not recommended.


## Structure
```
Audio_Aug_Lib |
              |-- Processing |
              |              |-- processor.py
              |              |-- augmentations.py
              |
              |-- Utils |
              |         |-- tools.py
              |         |-- signal_generators.py
              |         |-- decorators.py
              |
              |-- CLI_app.py
```
* `processor.py` contains the wrapper that sequentially executes all the augmentations.
* `augmentations.py` contains the implementations of the augmentation algorithms. Read chapter "Augmentations" for more details.
* `tools.py` contains the auxiliary functions for reading, saving, etc.
* `signal_generators.py` contains the functions for the generation of the signals (especially for colored noises).
* `decorators.py` contains the decorators mostly used for checking the correctness of functions arguments.



## Augmentations
The package contains the most comon augmentations used in speach recognition problem [[1](https://cyberleninka.ru/article/n/metody-augmentatsii-audio-signala/), [2](https://doi.org/10.21437/Interspeech.2019-2680)]
* **Time Stretch**: This is a common method applied in [similar problems](https://doi.org/10.1007/s10772-021-09883-3). It uses most common and quality/performance balanced solution - phase vocoder (stft based). The stft spectra is linearly interpolated over time, and phases are reconstructed too. Moreover, the transients are prevented for smoothing. The one was chosen because the previous tests with less complex [hann function](https://en.wikipedia.org/wiki/Hann_function) solution got worse results.


* **Pitch Shift**: The one simply resamples the time stretched signal. Since that uses phase vocoder in core.


* **Noise Addition**: Uses generators of colored noise (fft based) to generate and add noise to the signal. May be helpful to simulate digital devices noise and increase overall model generalization. 


* **Inverse**: Simply multiplies the signal by -1. Seems to be helpful 


* **Time Masking**: The [spectral augmentation](https://doi.org/10.21437/Interspeech.2019-2680) method introduced by google. The one masks some time frames of the signal with zeros, so the model should be more robust to the time shifts.


* **Frequency Masking**: One more [spectral augmentation](https://doi.org/10.21437/Interspeech.2019-2680) method that may prevent the model to generalize audio in the frequency domain. The one masks some frequency bins of the signal with zeros. Implemented by using stft transformation.


* **High/Low Pass**: Much faster in processing but less robust alternative to the frequency masking. The one simply cuts off the high or low frequencies of the signal. Implemented using the [butterworth filter](https://en.wikipedia.org/wiki/Butterworth_filter).




## How to Use CLI Application
The CLI application automatically applies all the transformations using recommended parameters. The one can process a single file or all the files in the chosen directory.
### Flags:
* -h, --help: show the help message and exit
* -s, --stereo: convert all the signals to stereo (mono by default)
### Example:
#### ! Remove `examples/test_mono_aug.wav` and `examples/test_stereo_aug.wav`, before regenerating examples !
Here is the example of usage for a single file.
```bash
python Audio_Aug_Lib/CLI_app.py examples/test_mono.wav
```

And the same for all the files in the directory.
```bash
python Audio_Aug_Lib/CLI_app.py examples
```
The last one will generate 2 new files: `test_mono_aug.wav` and `test_stereo_aug.wav`. 



## How to Use Processor
The Processor class can be used to create a custom pipelines or integrate the augmentation algorithms into your code. Note, that utilization of this lib is possible everywhere except the `Audio_Aug_Lib` folder.  

### Example:
First of all, you need to import the necessary modules and functions. The `partial` function will need further to create a pipeline more conveniently.  
```python 
from functools import partial

from Audio_Aug_Lib.Processing.processor import Processor
from Audio_Aug_Lib.Utils.tools import *
from Audio_Aug_Lib.Processing.augmentations import *
```

Then, read the audio file. The `read_audio` function returns sampling rate `fs` and the audio signal `x`. Here `stereo` parameter reduces any signal to common form.

```python
file_path = 'examples/test_mono.wav'
fs, x = read_audio(file_path, stereo=True)
```

Create a pipeline. The order of augmentations matters, so the shown below is semantically correct only for the human speach audio. Using `partial` function, you can pass only the parameters you want to fix or change default (read function documentation).    
```python
pipeline = [
    partial(time_stretch, factor=1.5),
    partial(pitch_shift, factor=1.5, semitones=True),
    partial(noise_addition, factor=0.1, noise_type="violet"),
    partial(inverse),
    partial(time_masking, factor=0.05, num_masks=3),
    partial(frequency_masking, fs=fs, factor=0.1, num_masks=3),
    partial(freq_pass, fs=fs, f_type='lowpass', cutoff=3000, poles=5)
]
```

Process the audio signal using the pipeline.  
```python
processor = Processor(pipeline)

x_aug = processor.process(x)
```

Save the augmented audio signal. Note that `save_audio` always normalizes the signal between -1 and 1. Also it is restricted to overwrite the existing files, so make sure you have a name different from original.  
```python
save_audio(x_aug, fs, 'examples/test_mono_aug.wav')
```




## Compatability
The package was tested on `Python 3.7`. Compatability with other versions is not guaranteed. Also, make sure you have installed the `requirements.txt` file.  
```bash
pip install -r requirements.txt
```

