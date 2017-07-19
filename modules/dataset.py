import os
import sys
import json
import collections
import numpy as np
from modules.audio_signal import compute_spectrograms
from modules.path import root_path
from utils import ProgressBar
import pickle

# create Datasets type
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

EOT_token = 0
PAD_token = 1


def indexes_from_text(lang, text):
    return [lang.char2index[char] for char in text]


def pad_time_dim(data, new_time, fill_constant):
    """data is a 2D numpy array"""
    assert new_time >= data.shape[1], \
        "new_time: %d, old_time: %d" % (new_time, data.shape[1])
    pad_time = new_time - data.shape[1]
    npad = ((0, 0), (0, pad_time))
    return np.pad(
        data,
        pad_width=npad,
        mode='constant',
        constant_values=fill_constant)


def pad_indexes(indexes, new_length, fill_constant):
    """indexes is a list of integers"""
    assert new_length >= len(indexes)
    indexes = np.array(indexes, dtype=np.int)
    pad_length = new_length - indexes.shape[0]
    npad = ((0, pad_length))
    return np.pad(
        indexes,
        pad_width=npad,
        mode='constant',
        constant_values=fill_constant)


class Lang:
    def __init__(self):
        self.char2index = {}
        self.index2char = {0: '^', 1: '$'}
        self.num_chars = 2

    def index_text(self, text):
        for char in text:
            self.index_char(char)

    def index_char(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.num_chars
            self.index2char[self.num_chars] = char
            self.num_chars += 1


class DataSet:
    VERSION = '0.1'

    def __init__(self, texts, audio_files,
                 max_text_length=30, max_audio_length=100):
        assert len(texts) == len(audio_files), \
            "The length of texts doesn't match with the length of audios."
        self._num_examples = len(texts)
        self._texts = texts
        self._audio_files = audio_files
        self._max_text_length = max_text_length
        self._max_audio_length = max_audio_length

        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._mels = []
        self._mags = []
        self._indexed_texts = []

        self._preprocess()

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def texts(self):
        return self._texts

    @property
    def audio_files(self):
        return self._audio_files

    @property
    def max_text_length(self):
        return self._max_text_length

    @property
    def max_audio_length(self):
        return self._max_audio_length

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def _preprocess(self):
        self.lang = Lang()
        for text in self._texts:
            self.lang.index_text(text)

        for text in self._texts:
            indexes = indexes_from_text(self.lang, text)
            indexes.append(EOT_token)
            padded_indexes = pad_indexes(
                indexes, self._max_text_length, PAD_token)
            self._indexed_texts.append(padded_indexes)

        self._indexed_texts = np.stack(self._indexed_texts, axis=0)

        bar = ProgressBar(len(self._audio_files) - 1, unit='')
        for (audio_files_read, audio_file) in enumerate(self._audio_files):
            # (n_mels, T), (1+n_fft/2, T)
            mel, mag = compute_spectrograms(audio_file)
            padded_mel = pad_time_dim(mel, self._max_audio_length, 0)
            padded_mag = pad_time_dim(mag, self._max_audio_length, 0)
            self._mels.append(padded_mel.transpose())
            self._mags.append(padded_mag.transpose())

            bar.update(audio_files_read)

        self._mels = np.stack(self._mels, axis=0)
        self._mags = np.stack(self._mags, axis=0)

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._mels = self._mels[perm]
            self._mags = self._mags[perm]
            self._indexed_texts = self._indexed_texts[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return (self._mels[start:end],
                self._mags[start:end],
                self._indexed_texts[start:end])


TINY_WORDS = '/mnt/nfs/dataset/tts/tiny-words-v0'

def tiny_words(max_text_length=20, max_audio_length=60,
               max_dataset_size=sys.maxsize):
    data_path = os.path.join(TINY_WORDS)
    meta_list = json.load(
        open(os.path.join(data_path, 'meta.json'), 'r'))
    texts = [x['text'] for x in meta_list]
    audios = [os.path.join(data_path, x['audio']) for x in meta_list]

    texts = texts[:max_dataset_size]
    audios = audios[:max_dataset_size]

    return DataSet(texts, audios,
                   max_text_length=max_text_length,
                   max_audio_length=max_audio_length,
                   preprocess=preprocess)


def make_lang(data_path, max_dataset_size=sys.maxsize):
    meta_list = json.load(
        open(os.path.join(data_path, 'meta.json'), 'r'))
    texts = [x['text'] for x in meta_list]
    texts = texts[:max_dataset_size]
    lang = Lang()
    for text in texts:
        lang.index_text(text)
    print(texts)
    return lang



def main():
    BATCH_SIZE = 32
    ds = tiny_words()
    print(ds.next_batch(BATCH_SIZE))

if __name__ == "__main__":
    main()
