import pandas as pd
import os

root_path = "./"


def extract_data_from_line(line):
    data = []
    artist_name_index = line.find(',')
    name = line[:artist_name_index].lower().replace("\"", "")
    # The replace between " to empty added beacuse some names contained it, probably error at the data creation
    line = line[artist_name_index + 1:]
    song_name_index = line.find(',')
    song_name = line[:song_name_index].lower()
    lyrics = line[song_name_index + 1:]
    more_than_one = lyrics.find('&  &  &')  # Indicator for 2 songs in same line, bad dataset :(
    if more_than_one != -1:
        curr_lyrics = lyrics[:more_than_one]
        data.append([name, song_name, curr_lyrics])
        return data + extract_data_from_line(lyrics[more_than_one + len('&  &  &'):])
    else:
        data.append([name, song_name, lyrics])
        return data


def create_lyrics_df_from(lyrics_file):
    train_lyrics_df = pd.DataFrame(columns=['name', 'song_name', 'lyrics'])
    with open(root_path + "/" + lyrics_file, 'r') as f:
        index = 0
        for line in f:
            data = extract_data_from_line(line)
            for row in data:
                train_lyrics_df.loc[index] = row
                index += 1

    print(len(train_lyrics_df))
    return train_lyrics_df


def create_midi_with_lyrics_df(train=True):
    if train:
        midi_path = os.path.join(root_path, "midi_files")
        lyrics_file = "lyrics_train_set.csv"
    else:
        midi_path = os.path.join(root_path, "midi_files/test")
        lyrics_file = "lyrics_test_set.csv"

    midi_files = [os.path.join(midi_path, path) for path in os.listdir(midi_path)
                  if '.mid' in path or '.midi' in path]

    train_midi_df = pd.DataFrame(columns=['name', 'song_name', 'midi_path'])

    for index, path in enumerate(midi_files):
        midi_name = path[path.find('midi_files/') + len('midi_files/'):].replace('_', ' ')
        sep_index = midi_name.find('-')
        name = midi_name[:sep_index - 1].lower()
        song_name = midi_name[sep_index + 2:midi_name.find('.mid')].lower()
        train_midi_df.loc[index] = [name, song_name, path]

    train_lyrics_df = create_lyrics_df_from(lyrics_file)
    train_df = pd.merge(train_midi_df, train_lyrics_df, how='right', on=['name', 'song_name'])
    # Merge on right (lyrics), because not all midi files have lyrics :(
    print(len(train_df))
