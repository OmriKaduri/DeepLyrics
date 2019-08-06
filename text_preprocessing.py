import string
import nltk
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def parse_raw_song(line):
    data = []
    artist_name_index = line.find(',')
    name = line[:artist_name_index].lower().replace("\"", "").strip()
    # The replace between " to empty added beacuse some names contained it, probably error at the data creation
    line_after_name = line[artist_name_index + 1:]
    song_name_index = line_after_name.find(',')
    song_name = line_after_name[:song_name_index].lower().strip()
    lyrics = line_after_name[song_name_index + 1:]
    more_than_one = lyrics.find('&  &  &')  # Indicator for 2 songs in same line, bad dataset :(
    if more_than_one != -1:
        curr_lyrics = lyrics[:more_than_one]
    else:
        curr_lyrics = lyrics
    curr_song = {'name': name, 'song_name': song_name, 'lyrics': curr_lyrics}
    if more_than_one != -1:
        data.append(curr_song)
        more_songs = parse_raw_song(lyrics[more_than_one + len('&  &  &'):])
        for song in more_songs:
            data.append(song)
        return data
    else:
        return [curr_song]

def remove_brackets(lyrics):
    while lyrics.find('(') != -1:
        open_bracket = lyrics.find('(')
        close_bracket = lyrics.find(')')
        lyrics = lyrics[:open_bracket] + lyrics[close_bracket + 1:]
    return lyrics

def parse_raw_songs(raw_songs):
    songs = []

    for raw_song in raw_songs:
        curr_songs = parse_raw_song(raw_song)  # Iterate beacuse might contain multiple songs in a raw song line
        for song in curr_songs:
            songs.append(song)

    return songs


def song_midi_filename(song):
    return song['name'].replace(' ', '_') + "_-_" + song['song_name'].replace(' ', '_')

def all_midi_files(midi_dir):
    midi_path = os.path.join('./', midi_dir)

    return [os.path.join(midi_path, path) for path in os.listdir(midi_path)
            if '.mid' in path or '.midi' in path]

def create_songs_for(train=True):
    root_path = './'
    if train:
        midi_dir = "midi_files"
        lyrics_file = "lyrics_train_set.csv"
    else:
        midi_dir = "midi_files/test"
        lyrics_file = "lyrics_test_set.csv"

    with open(root_path + "/" + lyrics_file, 'r') as raw_lyrics:
        raw_songs = raw_lyrics.read().splitlines()

    songs = parse_raw_songs(raw_songs)

    midi_files = all_midi_files(midi_dir)

    for i, song in enumerate(songs):
        midi_name = song_midi_filename(song)

        matched_midi_files = [midi_file for midi_file in midi_files
                              if midi_name in midi_file.lower()]

        if len(matched_midi_files) != 1:
            print("OH OH", len(matched_midi_files), song)
            continue

        songs[i]['midi_file'] = matched_midi_files[0]

        if songs[i]['lyrics'].find('&,,,,') == -1:
            songs[i]['lyrics'] = songs[i]['lyrics'] + ' EOS'
        else:
            songs[i]['lyrics'] = songs[i]['lyrics'].replace('&,,,,', ' EOS')

        songs[i]['lyrics'] = ' '.join(nltk.word_tokenize(songs[i]['lyrics']))
        songs[i]['lyrics'] = songs[i]['lyrics'].replace('&', 'EOL')

        songs[i]['lyrics'] = remove_brackets(songs[i]['lyrics'])

        songs[i]['lyrics'] = '<start> ' + songs[i]['lyrics']
        # splitted_lyrics = [token for token in nltk.word_tokenize(songs[i]['lyrics']) if token not in string.punctuation]
        
        # songs[i]['splitted_lyrics'] = splitted_lyrics

    return songs

def tokenize_lyrics(lyrics):
  tokenizer = Tokenizer(filters='')
  tokenizer.fit_on_texts(lyrics)
  sequences = tokenizer.texts_to_sequences(lyrics)

  return tokenizer, sequences

def create_embedding_matrix(dictionary):
  embeddings_index = {}
  f = open('glove.6B.300d.txt', encoding="utf8")
  EMBEDDING_DIM = 300
  for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
  f.close()

  embedding_matrix = np.zeros((len(dictionary) + 1, EMBEDDING_DIM))
  unk_words = []
  for word, i in dictionary.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
          # words not found in embedding index will be all-zeros.
          embedding_matrix[i] = embedding_vector
      else:
          unk_words.append(word)

  print("Num of unknown words:", len(unk_words))

  return embedding_matrix

def preprocess_texts(lyrics, seq_len=600):
  tokenizer, sequences = tokenize_lyrics(lyrics)
  padded_seqs = pad_sequences(sequences, seq_len)
  embedding_matrix = create_embedding_matrix(tokenizer.word_index)

  return tokenizer, padded_seqs, embedding_matrix
