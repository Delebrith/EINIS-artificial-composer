import datetime
import numpy as np
import pypianoroll
from matplotlib import pyplot as plt
from keras.layers import Dense, LeakyReLU, LSTM, BatchNormalization, \
    Input, Bidirectional, Conv2D, Conv2DTranspose, Reshape
from keras.models import Sequential, Model
from keras.optimizers import Adam
from tensorflow.python.ops.init_ops import RandomNormal

from scripts.DataLoader import DataLoader
from scripts.pianorolldcgan.PianoRollDCGAN import PianoRollDCGAN


class PianoRollLstmGAN(PianoRollDCGAN):
    def __init__(self, dataloader: DataLoader, g_lr=0.001, g_beta=0.999, d_lr=0.001, d_beta=0.999, latent_dim=1024,
                 content_shape=(128, 128, 1)):
        self.seq_length = dataloader.features
        self.seq_shape = (128, self.seq_length)
        PianoRollDCGAN.__init__(self=self, dataloader=dataloader, latent_dim=latent_dim,
                                content_shape=content_shape, g_lr=g_lr, d_lr=d_lr, g_beta=g_beta, d_beta=d_beta)
        self.name = "pianoroll-LSTM-GAN"

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

        model.add(Conv2D(1, (7, 7), activation='sigmoid', padding='same', kernel_initializer=RandomNormal(stddev=0.5)))
        model.add(Reshape(self.seq_shape))

        return model

    def build_discriminator(self, lr=0.01, beta=0.9):
        model = Sequential()
        model.add(
            LSTM(256, input_shape=self.seq_shape, return_sequences=True, kernel_initializer=RandomNormal(stddev=0.5)))
        model.add(Bidirectional(LSTM(256, kernel_initializer=RandomNormal(stddev=0.5))))

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
        discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr, beta_1=beta),
                              metrics=['accuracy'])
        return discriminator

    def generate_sample(self, epoch):
        path = "../samples/%s_%s_epoch_%d" % (datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), self.name, epoch)
        generated = self.generator.predict(np.random.randn(128, self.latent_dim))
        generated = generated.reshape(128, 128, 1)
        plt.imshow(generated)
        plt.savefig("%s.png" % path)
        plt.close()

        generated *= 128.0 / generated.max()

        track = pypianoroll.StandardTrack(name='drums', is_drum=False, pianoroll=generated, program=0)
        multitrack = pypianoroll.Multitrack(
            name='multitrack',
            tracks=[track],
            tempo=np.asarray([[8.0] for _ in range(128)]),
            resolution=24
        )
        multitrack.write(path=("%s.mid" % path))
