import logging
import os
import numpy as np

from mido import MidiFile

from scripts.DataLoader import DataLoader


class MidiToImgDataLoader(DataLoader):
    def __init__(self, path, features):
        DataLoader.__init__(self=self, path=path, features=features)

    def get_batch(self, batch_num: int, batch_size: int):
        batch = self.data[batch_num * batch_size:(batch_num + 1) * batch_size]
        return np.array(batch).reshape(batch_size, len(batch[0]), len(batch[0][0]), 1)

    def read_one_file(self, filename):
        try:
            logging.info("Reading %s" % filename)
            notes = self.read_notes(os.path.join(self._path, filename))
            num_of_fragments = int(len(notes) / self.features)
            fragments = [notes[self.features * (i - 1):self.features * i] for i in range(1, num_of_fragments)]
            self.data = self.data + fragments
        except:
            print('Error reading {}'.format(os.path.join(self._path, filename)))
            return None

    def read_notes(self, path):
        mid = MidiFile(path)
        notes = []
        for msg in mid:
            if not msg.is_meta and msg.channel == 0 and msg.type == 'note_on':
                data = msg.bytes()
                notes.append(self.read_as_array(data[1]))
        return np.array(notes)

    def read_as_array(self, param):
        array = np.zeros(self.features)
        array[param] = 1
        return array

    def get_number_of_batches(self, batch_size: int):
        return int(len(self.data)/batch_size)
