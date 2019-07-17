import gensim
import numpy as np

from data import create_midi_with_lyrics_df
from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential

import tensorflow as tf
import sys


class LyricsLangModel:

    def __init__(self, df):
        self.df = df
        self.word_model = {}
        self.pretrained_weights = {}
        self.vocab_size = -1
        self.emdedding_size = -1
        self.train_x = []
        self.train_y = []
        self.unk_words = 0

    def load_word2vec_model(self, file_path="GoogleNews-vectors-negative300.bin"):
        self.word_model = gensim.models.KeyedVectors.load_word2vec_format(file_path,
                                                                          binary=True)
        max_lyrics_size = 140
        words = []
        for i in range(len(self.df)):
            words.append([word for word in self.df['lyrics'][i].split()[:max_lyrics_size]])

        # word_model = gensim.models.Word2Vec(words, size=100, min_count=1, window=5, iter=100)
        self.pretrained_weights = self.word_model.wv.syn0
        self.vocab_size, self.emdedding_size = self.pretrained_weights.shape
        print('Result embedding shape:', self.pretrained_weights.shape)
        print('Checking similar words:')
        for word in ['small', 'child', 'love', 'goodbye']:
            most_similar = ', '.join(
                '%s (%.2f)' % (similar, dist) for similar, dist in self.word_model.most_similar(word)[:5])
            print('  %s -> %s' % (word, most_similar))

        self.train_x = np.zeros([len(words), max_lyrics_size], dtype=np.int32)
        self.train_y = np.zeros([len(words)], dtype=np.int32)
        for i, sentence in enumerate(words):
            for t, word in enumerate(sentence[:-1]):
                self.train_x[i, t] = self.word2idx(word)
            self.train_y[i] = self.word2idx(sentence[-1])
        print('train_x shape:', self.train_x.shape)
        print('train_y shape:', self.train_y.shape)
        print("Number of unknown words:", self.unk_words)

    def word2idx(self, word):
        if word in self.word_model.wv.vocab:
            return self.word_model.wv.vocab[word].index
        elif word.upper() in self.word_model.wv.vocab:
            return self.word_model.wv.vocab[word.upper()].index
        else:
            self.unk_words = self.unk_words + 1
            return self.word_model.wv.vocab['UNK'].index

    def idx2word(self, idx):
        return self.word_model.wv.index2word[idx]

    def sample(self, preds, temperature=1.0):
        if temperature <= 0:
            return np.argmax(preds)
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate_next(self, text, num_generated=10):
        word_idxs = [self.word2idx(word) for word in text.lower().split()]
        for i in range(num_generated):
            prediction = self.model.predict(x=np.array(word_idxs))
            idx = self.sample(prediction[-1], temperature=0.7)
            word_idxs.append(idx)
        return ' '.join(self.idx2word(idx) for idx in word_idxs)

    def on_epoch_end(self, epoch, _):
        print('\nGenerating text after epoch: %d' % epoch)
        texts = [
            'close',
            'if',
            'dear',
            'all',
        ]
        for text in texts:
            sample = self.generate_next(text)
            print('%s... -> %s' % (text, sample))

    def train(self):
        with tf.device('/gpu:0'):
            model = Sequential()
            model.add(
                Embedding(input_dim=self.vocab_size, output_dim=self.emdedding_size, weights=[self.pretrained_weights]))
            model.add(LSTM(units=self.emdedding_size))
            model.add(Dense(units=self.vocab_size))
            model.add(Activation('softmax'))
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

            model.fit(self.train_x, self.train_y,
                      batch_size=128,
                      epochs=20,
                      callbacks=[LambdaCallback(on_epoch_end=self.on_epoch_end)])


train_df = create_midi_with_lyrics_df()

print("Loaded training lyrics successfully.")
lang_model = LyricsLangModel(train_df)
if len(sys.argv) > 1:
    print("Loading word2vec model file from", sys.argv[1])
    lang_model.load_word2vec_model(sys.argv[1])
else:
    lang_model.load_word2vec_model()

print("Loaded Google word2vec model successfully.")
lang_model.train()
