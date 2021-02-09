import datetime

import numpy as np
from keras.layers import Dense, LeakyReLU, Reshape, LSTM, BatchNormalization, \
    Input, Bidirectional
from keras.models import Sequential, Model
from keras.optimizers import Adam
from mido import MidiFile, MidiTrack, Message
from tensorflow.python.ops.init_ops import RandomNormal

from scripts.DataLoader import DataLoader
from scripts.GAN import GAN


class MidiLSTMGan(GAN):
    def __init__(self, dataloader: DataLoader = None, g_lr=0.001, g_beta=0.999, d_lr=0.001, d_beta=0.999, latent_dim=512):
        GAN.__init__(self=self, latent_dim=latent_dim, data_generator=dataloader, name="midi-notes-lstm-gan-scalec")
        self.data_loader = dataloader
        self.seq_length = dataloader.features
        self.seq_shape = (1, self.seq_length)
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator(lr=d_lr, beta=d_beta)
        self.combined = self.combined_model(lr=g_lr, beta=g_beta)

    def build_discriminator(self, lr=0.01, beta=0.9):
        model = Sequential()
        model.add(
            LSTM(128, input_shape=self.seq_shape, return_sequences=True, kernel_initializer=RandomNormal(stddev=0.5)))
        model.add(Bidirectional(LSTM(512, kernel_initializer=RandomNormal(stddev=0.5))))

        model.add(Dense(256, kernel_initializer=RandomNormal(stddev=0.5)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(128, kernel_initializer=RandomNormal(stddev=0.5)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(64, kernel_initializer=RandomNormal(stddev=0.5)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1, activation='sigmoid', kernel_initializer=RandomNormal(stddev=0.5)))
        model.summary()

        seq = Input(shape=self.seq_shape)
        validity = model(seq)

        discriminator = Model(seq, validity)
        discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr, beta_1=beta), metrics=['accuracy'])
        return discriminator

    def build_generator(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim, kernel_initializer=RandomNormal(stddev=0.5)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(256, input_dim=self.latent_dim, kernel_initializer=RandomNormal(stddev=0.5)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512, kernel_initializer=RandomNormal(stddev=0.5)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.seq_shape), activation='sigmoid', kernel_initializer=RandomNormal(stddev=0.5)))
        model.add(Reshape(self.seq_shape))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        seq = model(noise)

        return Model(noise, seq)

    def combined_model(self, lr=0.001, beta=0.999):
        self.discriminator.trainable = False
        model = Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        opt = Adam(lr=lr, beta_1=beta)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    def generate_sample(self, epoch):
        path = "../samples/%s_%s_epoch_%d.mid" \
               % (datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), self.name, epoch)
        self.generate_sample_to(path=path)

    def generate_sample_to(self, path):
        generated = self.generator.predict(np.random.randn(1, self.latent_dim))
        generated = generated.reshape(self.data_loader.features)
        mid = MidiFile()
        track = MidiTrack()
        t = 0
        for note in generated:
            note = int(127.0 * note)
            msg = Message('note_on', note=note)
            t = t + 1
            msg.time = t
            msg.velocity = 32
            track.append(msg)
        mid.tracks.append(track)
        mid.save(path)
