import pandas as pd
import os
import string
import nltk

root_path = "./"


def extract_data_from_line(line):
    data = []
    artist_name_index = line.find(',')
    name = line[:artist_name_index].lower().replace("\"", "").strip()
    # The replace between " to empty added beacuse some names contained it, probably error at the data creation
    line = line[artist_name_index + 1:]
    song_name_index = line.find(',')
    song_name = line[:song_name_index].lower().strip()
    lyrics = line[song_name_index + 1:]
    more_than_one = lyrics.find('&  &  &')  # Indicator for 2 songs in same line, bad dataset :(
    curr_lyrics = lyrics[:more_than_one]
    data.append([name, song_name, curr_lyrics])
    if more_than_one != -1:
        return data + extract_data_from_line(lyrics[more_than_one + len('&  &  &'):])
    else:
        return data


def create_lyrics_df_from(lyrics_file):
    lyrics_df = pd.DataFrame(columns=['name', 'song_name', 'lyrics'])
    lyrics_df = lyrics_df.astype(dtype={'name': str,
                                        'song_name': str,
                                        'lyrics': str})
    with open(root_path + "/" + lyrics_file, 'r') as f:
        index = 0
        for line in f:
            data = extract_data_from_line(line)
            for row in data:
                lyrics_df.loc[index] = row
                index += 1

    print(len(lyrics_df))
    return lyrics_df


def create_midi_with_lyrics_df(train=True):
    if train:
        midi_dir = "midi_files"
        lyrics_file = "lyrics_train_set.csv"
    else:
        midi_dir = "midi_files/test"
        lyrics_file = "lyrics_test_set.csv"

    midi_path = os.path.join(root_path, midi_dir)
    midi_files = [os.path.join(midi_path, path) for path in os.listdir(midi_path)
                  if '.mid' in path or '.midi' in path]

    midi_df = pd.DataFrame(columns=['name', 'song_name', 'midi_path'])
    midi_df = midi_df.astype(dtype={'name': str,
                                    'song_name': str,
                                    'midi_path': str})
    for index, path in enumerate(midi_files):
        midi_name = path[path.find(midi_path) + len(midi_path) + 1:].replace('_', ' ')
        sep_index = midi_name.find('-')
        name = midi_name[:sep_index - 1].lower().strip()
        song_name = midi_name[sep_index + 2:midi_name.find('.mid')].lower().strip()
        midi_df.loc[index] = [name, song_name, path]

    lyrics_df = create_lyrics_df_from(lyrics_file)
    df = pd.merge(midi_df, lyrics_df, how='inner', on=['name', 'song_name'])
    # Merge on inner, because not all midi files have lyrics and vice versa     :(
    print(len(df))
    return preprocess_lyrics(df)


def preprocess_lyrics(df):
    processed_lyrics = []
    for index, row in df.iterrows():
        lyrics = row['lyrics'].replace('&', '<\s>').lstrip()
        # lyrics = '<BOS> ' + lyrics
        if lyrics.find(',,,,') == -1:
            lyrics = lyrics + '<EOS>'
        else:
            lyrics = lyrics.replace(',,,,', ' <EOS>')

        while lyrics.find('(') != -1:
            open_bracket = lyrics.find('(')
            close_bracket = lyrics.find(')')
            lyrics = lyrics[:open_bracket] + lyrics[close_bracket + 1:]
        #       Removing brackets, currently regarded as noise to the lyrics

        #     for sentence in lyrics.split(' '):
        #       print(sentence)
        row['lyrics'] = lyrics
        processed_lyrics.extend([w for w in nltk.word_tokenize(lyrics)])

    return processed_lyrics

