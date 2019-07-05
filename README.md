# DeepLyrics

## Data preparation ##
The data splitted to train and test, each consists of two parts:
1. Midi files - under `midi_files` directory
2. lyrics `lyrics_MODE_set.csv` (where `MODE` is train/test)

At `data.py` the path of the directory under which the directory and the lyric files located should be set to the variable `root_path`.

**Important** to note that the original midi data wasn't splitted to test and train. This is done for preprocessing convenience.
