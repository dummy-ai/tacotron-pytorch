import collections
import numpy as np
from modules.melscale import melscale

# create Datasets type
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

EOT_token = 0
PAD_token = 1

def indexes_from_text(lang, text):
    return [lang.char2index[char] for char in text]

def pad_time_dim(data, new_time, fill_constant):
    """data is a 2D numpy array"""
    assert new_time >= data.shape[1]
    pad_time = new_time - data.shape[1]
    npad = ((0, 0), (0, pad_time))
    return np.pad(data,
        pad_width=npad, 
        mode='constant', 
        constant_values=fill_constant)

def pad_indexes(indexes, new_length, fill_constant):
    """indexes is a list of integers"""
    assert new_length >= len(indexes)
    indexes = np.array(indexes, dtype=np.int)
    pad_length = new_length - indexes.shape[0]
    npad = ((0, pad_length))
    return np.pad(indexes, 
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
        self._spectros = []
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
    def epochs_completed(self):
        return self._epochs_completed

    def _preprocess(self):
        self.lang = Lang()
        for text in self._texts:
            self.lang.index_text(text)

        for text in self._texts:
            indexes = indexes_from_text(self.lang, text)
            indexes.append(EOT_token)
            padded_indexes = pad_indexes(indexes,
                self._max_text_length, PAD_token)
            self._indexed_texts.append(padded_indexes)

        self._indexed_texts = np.stack(self._indexed_texts, axis=0)

        for audio_file in self._audio_files:
            mel_spectro = melscale(audio_file)
            padded_mel_spectro = pad_time_dim(
                mel_spectro, self._max_audio_length, 0)
            self._spectros.append(padded_mel_spectro)

        self._spectros = np.stack(self._spectros, axis=0)

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._spectros = self._spectros[perm]
            self._indexed_texts = self._indexed_texts[perm]
            # Start next epoch 
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._spectros[start:end], self._indexed_texts[start:end]

def main():
    import json
    import os
    from modules.path import root_path
    data_path = os.path.join(root_path, 
        'data/tiny-words-v0/')
    meta_list = json.load(
        open(os.path.join(data_path, 'meta.json'), 'r'))
    texts = [x['text'] for x in meta_list]
    audios = [os.path.join(data_path, x['audio']) for x in meta_list]
    ds = DataSet(texts, audios,
        max_text_length=20, max_audio_length=30)
    BATCH_SIZE = 32
    print(ds.next_batch(BATCH_SIZE))

if __name__ == "__main__":
    main()
