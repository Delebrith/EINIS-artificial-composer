import datetime

import numpy as np
import pypianoroll
from keras.layers import Dense, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, Dropout, Flatten, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from tensorflow.python.ops.init_ops import RandomNormal

from scripts.DataLoader import DataLoader
from scripts.GAN import GAN


class PianoRollDCGAN(GAN):
    def __init__(self, dataloader: DataLoader = None, g_lr=0.001, g_beta=0.999, d_lr=0.001, d_beta=0.999, latent_dim=1024,
                 content_shape=(128, 128, 3)):
        GAN.__init__(self=self, data_generator=dataloader, name="pianoroll-DC-GAN", latent_dim=latent_dim,
                     content_shape=content_shape)
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator(lr=d_lr, beta=d_beta)
        self.combined = self.combined_model(lr=g_lr, beta=g_beta)

    def build_generator(self):
        model = Sequential()
        # foundation for 8x8 image
        n_nodes = 128 * 8 * 8
        model.add(Dense(n_nodes, input_dim=self.latent_dim, kernel_initializer=RandomNormal(stddev=0.5)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((8, 8, 128)))

        # upsample to 16X16
        model.add(
            Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=0.5)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # upsample to 32x32
        model.add(
            Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=0.5)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # upsample to 64x64
        model.add(
            Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=0.5)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # upsample to 128x128
        model.add(
            Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=0.5)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(3, (7, 7), activation='sigmoid', padding='same', kernel_initializer=RandomNormal(stddev=0.5)))
        return model

    def build_discriminator(self, lr=0.000001, beta=0.999):
        model = Sequential()
        model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same', input_shape=self.content_shape,
                         kernel_initializer=RandomNormal(stddev=0.5)))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=0.5)))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=0.5)))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=0.5)))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid', kernel_initializer=RandomNormal(stddev=0.5)))

        # compile model
        opt = Adam(lr=lr, beta_1=lr)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def combined_model(self, lr=0.0001, beta=0.999):
        self.discriminator.trainable = False
        model = Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        opt = Adam(lr=lr, beta_1=beta)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    def generate_sample(self, epoch):
        path = "../samples/%s_%s_epoch_%d" % (datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), self.name, epoch)
        self.generate_sample_to(path=path)

    def generate_sample_to(self, path):
        generated = self.generator.predict(np.random.randn(1, self.latent_dim))
        generated = generated.reshape(128, 128, 3)
        plt.imshow(generated)
        plt.savefig("%s.png" % path)
        plt.close()

        generated *= 128.0 / generated.max()
        generated = generated.astype('int32')
        generated = generated.astype('float64')

        drums = generated[:, :, 0]
        melody1 = generated[:, :, 1]
        melody2 = generated[:, :, 2]

        track0 = pypianoroll.StandardTrack(name='drums', is_drum=True, pianoroll=drums, program=0)
        track1 = pypianoroll.StandardTrack(name='piano1', is_drum=False, pianoroll=melody1, program=0)
        track2 = pypianoroll.StandardTrack(name='piano2', is_drum=False, pianoroll=melody2, program=52)
        multitrack = pypianoroll.Multitrack(
            name='multitrack',
            tracks=[track0, track1, track2],
            tempo=np.asarray([[8.0] for _ in range(128)]),
            resolution=24
        )
        multitrack.write(path=("%s.mid" % path))

