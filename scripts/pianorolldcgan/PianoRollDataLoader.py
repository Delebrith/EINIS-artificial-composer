import logging
import os
import pypianoroll

from scripts.DataLoader import DataLoader

import numpy as np


class PianoRollDataLoader(DataLoader):
    def __init__(self, path, features, augmentation=False):
        self.augmentation = augmentation
        DataLoader.__init__(self, path=path, features=features)

    def get_batch(self, batch_num: int, batch_size: int):
        batch = self.data[batch_num * batch_size:(batch_num + 1) * batch_size]
        batch = np.array(batch).reshape(batch_size, 128, 128, 3)
        batch += np.random.uniform(low=0.0, high=0.01, size=batch.shape)
        return batch

    def read_one_file(self, filename):
        try:
            logging.info("Reading %s" % filename)
            notes = self.read_notes(os.path.join(self._path, filename))
            if self.augmentation is False:
                num_of_fragments = int(len(notes) / self.features)
                fragments = [
                    notes[self.features * (i - 1):self.features * i]
                    for i in range(1, num_of_fragments)
                ]
                self.data = self.data + fragments
            else:
                fragments = []
                start = 0
                end = self.features
                while end < len(notes):
                    fragments.append(notes[start:end])
                    start += 8
                    end += 8

                self.data = self.data + fragments
        except:
            print('Error reading {}'.format(os.path.join(self._path, filename)))
            return None

    def read_notes(self, path):
        mid = pypianoroll.read(path=path)
        length = max([len(track.pianoroll) for track in mid.tracks])

        drums = [track.pianoroll for track in mid.tracks if track.is_drum]
        drums = sum(drums) if len(drums) > 0 else np.zeros((length, 128))
        drums = drums[0:length:8]

        melody = [track for track in mid.tracks if not track.is_drum]
        melody1 = [track.pianoroll for track in melody if track.program < 50]
        melody1 = sum(melody1) if len(melody1) > 0 else np.zeros((length, 128))
        melody1 = melody1[0:length:8]
        melody2 = [track.pianoroll for track in melody if track.program >= 50]
        melody2 = sum(melody2) if len(melody2) > 0 else np.zeros((length, 128))
        melody2 = melody2[0:length:8]

        notes = np.zeros((drums.shape[0], 128, 3))
        notes[:, :, 0] = drums
        notes[:, :, 1] = melody1
        notes[:, :, 2] = melody2

        notes *= 1.0 / notes.max()

        return notes


    def read_as_array(self, param):
        array = np.zeros(self.features)
        array[param] = 1
        return array

    def get_number_of_batches(self, batch_size: int):
        return int(len(self.data)/batch_size)
