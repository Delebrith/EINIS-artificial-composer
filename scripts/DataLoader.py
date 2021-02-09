import logging
import os
import random


class DataLoader:
    def __init__(self, path, features):
        self._path = path
        self.features = features
        self.data = []
        if self._path is not None:
            self.read_data()

    def read_data(self):
        [self.read_one_file(file) for file in os.listdir(self._path)]
        logging.info("data read successfully. %d samples" % len(self.data))

    def get_batch(self, batch_num: int, batch_size: int):
        raise NotImplemented

    def get_number_of_batches(self, batch_size: int):
        raise NotImplemented

    def read_one_file(self, file):
        raise NotImplemented

    def shuffle_samples(self):
        random.shuffle(self.data)
