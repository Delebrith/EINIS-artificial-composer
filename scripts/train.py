import argparse

import numpy as np
import logging
import tensorflow as tf

from scripts.GAN import GAN
from scripts.lstmdisc.MidiLstmGAN import MidiLSTMGan
from scripts.lstmdisc.MidiToSequenceDataLoader import MidiToSequenceDataLoader
from scripts.pianorolldcgan.PianoRollDCGAN import PianoRollDCGAN
from scripts.pianorolldcgan.PianoRollDataLoader import PianoRollDataLoader
from scripts.simplecnn.MidiToImgDataLoader import MidiToImgDataLoader
from scripts.simplecnn.SimpleCnnGAN import SimpleCnnGAN
from sys import exit

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8),
                                  device_count={'GPU': 1})
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

tf.compat.v1.disable_v2_behavior()
physical_devices = tf.compat.v1.config.experimental.list_physical_devices('GPU')
tf.compat.v1.config.experimental.set_memory_growth(physical_devices[0], True)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def train(gan: GAN, batch_size: int, epochs: int, pretrain_generator_epochs: int, sampling_rate: int):
    generator = gan.generator
    discriminator = gan.discriminator
    combined = gan.combined
    data_loader = gan.data_loader

    for e in range(pretrain_generator_epochs):
        logging.info('pretraining generator epoch %d of %d', e, pretrain_generator_epochs)
        data_loader.shuffle_samples()
        batches = data_loader.get_number_of_batches(batch_size)
        for b in range(batches):
            x, y = np.random.randn(batch_size, gan.latent_dim), np.ones(batch_size)
            pre_g_loss = combined.train_on_batch(x, y)
            # print("Batch %d of %d - g_loss: %f" % (b, batches, pre_g_loss),
            #       end=('\r' if b != batches else '\n'))
        logging.info('pretraining loss: %f', pre_g_loss)

    for e in range(epochs):
        logging.info('training epoch %d of %d', e, epochs)
        data_loader.shuffle_samples()

        epoch_d_loss = []
        epoch_d_acc = []
        epoch_g_loss = []

        batches = data_loader.get_number_of_batches(batch_size)
        for b in range(batches):
            x_real, y_real = data_loader.get_batch(batch_num=b, batch_size=batch_size), np.ones(batch_size)
            d_loss_real, d_acc_real = discriminator.train_on_batch(x_real, y_real)

            x_fake, y_fake = generator.predict(np.random.randn(batch_size, gan.latent_dim)), np.zeros(batch_size)
            d_loss_fake, d_acc_fake = discriminator.train_on_batch(x_fake, y_fake)

            epoch_d_loss.append((d_loss_real + d_loss_fake) / 2)
            epoch_d_acc.append((d_acc_real + d_acc_fake) / 2)

        for b in range(batches):
            x_gan, y_gan = np.random.randn(batch_size, gan.latent_dim), np.ones(batch_size)
            g_loss = combined.train_on_batch(x_gan, y_gan)

            epoch_g_loss.append(g_loss)
            # print("Batch %d of %d - d_loss: %f, d_acc: %f, g_loss: %f"
            #       % (b, batches, epoch_d_loss[-1], epoch_d_acc[-1], epoch_g_loss[-1]),
            #       end=('\r' if b != batches-1 else '\n'))

        gan.d_loss.append(sum(epoch_d_loss))
        gan.d_acc.append(sum(epoch_d_acc) / len(epoch_d_acc))
        gan.g_loss.append(sum(epoch_g_loss))
        logging.info("d_loss: %f, d_acc: %f, g_loss: %f", gan.d_loss[-1], gan.d_acc[-1], gan.g_loss[-1])

        if e % sampling_rate == 0:
            gan.generate_sample(epoch=e)
            gan.save_models(e)
            gan.plot_progress("epoch_%d" % e)

    gan.plot_progress(suffix='final')


def main():
    parser = argparse.ArgumentParser(description="Training arg parser")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--pretraining_epochs', type=int, default=5)
    parser.add_argument('--sampling_rate', type=int, default=10)
    parser.add_argument('--load_generator_path', type=str, default=None)
    parser.add_argument('--load_discriminator_path', type=str, default=None)
    parser.add_argument('--gan_type', type=str, default='simple_cnn')
    parser.add_argument('--dataset', type=str, default='../../project/data/maestro-v2.0.0-midi/2018')
    args = parser.parse_args()

    dataset = args.dataset
    gan_type = args.gan_type

    if gan_type == 'simple_cnn':
        logging.info("selected type %s", gan_type)
        dataloader = MidiToImgDataLoader(features=128, path=dataset)
        gan = SimpleCnnGAN(dataloader=dataloader, g_lr=0.00001, g_beta=0.5, d_lr=0.000001, d_beta=0.5)
    elif gan_type == 'sequence_lstm':
        logging.info("selected type %s", gan_type)
        dataloader = MidiToSequenceDataLoader(features=128, path=dataset)
        # gan = MidiLSTMGan(dataloader=dataloader, d_lr=0.0005, g_lr=0.00001, g_beta=0.6, d_beta=0.9)
        gan = MidiLSTMGan(dataloader=dataloader, d_lr=0.0005, g_lr=0.000005, g_beta=0.9, d_beta=0.9)
    elif gan_type == 'piano_roll_cnn':
        logging.info("selected type %s", gan_type)
        dataloader = PianoRollDataLoader(path=dataset, features=128, augmentation=True)
        gan = PianoRollDCGAN(dataloader=dataloader, d_lr=0.0001, g_lr=0.0002, g_beta=0.6, d_beta=0.6)
    else:
        gan = None

    train(gan=gan, batch_size=args.batch_size, epochs=args.epochs, pretrain_generator_epochs=args.pretraining_epochs,
          sampling_rate=args.sampling_rate)


if __name__ == '__main__':
    main()
    exit(0)
