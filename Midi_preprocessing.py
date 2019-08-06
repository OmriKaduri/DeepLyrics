
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pretty_midi
from tqdm import tqdm
import numpy as np

def encode_midi(midi_files, fs=1):
  encodings = []

  bad_midis = ['./midi_files/Beastie_Boys_-_Girls.mid',
              './midi_files/Dan_Fogelberg_-_Leader_of_the_Band.mid',
              './midi_files/Brian_McKnight_-_On_The_Down_Low.mid',
              './midi_files/Aaron_Neville_-_Tell_It_Like_It_Is.mid',
              "./midi_files/Billy_Joel_-_Movin'_Out.mid",
              './midi_files/David_Bowie_-_Lazarus.mid',
              './midi_files/Billy_Joel_-_Pressure.mid']

  for file in tqdm(midi_files):
      if file in bad_midis:
          continue

      midi = pretty_midi.PrettyMIDI(file)
      piano_rolls = []
      max_seq = 0
      for inst in midi.instruments:
          if not inst.is_drum:
              roll = inst.get_piano_roll(fs=fs).T
              if roll.shape[0] > max_seq:
                  max_seq = roll.shape[0]
              piano_rolls.append(roll)
      song = np.zeros((max_seq, 12))
      for seq in range(max_seq):
          for roll in piano_rolls:
              try:
                  timestep = roll[seq, :] 
                  for i, value in enumerate(timestep):
                      if value > 0:
                          song[seq][i % 12] = 1
              except:
                  pass
      encodings.append(song)
          
      corpus = []

      for i,song in enumerate(encodings):
          words = []
          for j, seq in enumerate(song):
              words.append(''.join(str(e) for e in seq.astype(int)))
          corpus.append(' '.join(words))

  return corpus

def tokenize_midis(midi_encodings):
  midi_tokenizer = Tokenizer(filters='')
  midi_tokenizer.fit_on_texts(midi_encodings)
  seqs = midi_tokenizer.texts_to_sequences(midi_encodings)

  return midi_tokenizer, seqs

def preprocess_midi(songs, seq_len=300, fs=1):
  encoded_midis = encode_midi(songs, fs=fs)
  tokenizer, tokenized_midis = tokenize_midis(encoded_midis)
  padded_seqeunces = pad_sequences(tokenized_midis, maxlen=seq_len)

  return tokenizer, padded_seqeunces
