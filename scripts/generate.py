import argparse
import logging
import tensorflow as tf

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


def main():
    parser = argparse.ArgumentParser(description="Training arg parser")
    parser.add_argument('--load_generator_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--gan_type', type=str, default='simple_cnn')
    args = parser.parse_args()

    gan_type = args.gan_type

    if gan_type == 'simple_cnn':
        logging.info("Selected type %s", gan_type)
        dataloader = MidiToImgDataLoader(features=128, path=None)
        gan = SimpleCnnGAN(dataloader=None)
    elif gan_type == 'sequence_lstm':
        logging.info("Selected type %s", gan_type)
        dataloader = MidiToSequenceDataLoader(features=128, path=None)
        gan = MidiLSTMGan(dataloader=dataloader)
    elif gan_type == 'piano_roll_cnn':
        logging.info("Selected type %s", gan_type)
        dataloader = PianoRollDataLoader(path=None, features=128, augmentation=True)
        gan = PianoRollDCGAN(dataloader=None)
    else:
        gan = None

    logging.info("Loaded data from %s", args.load_generator_path)
    gan.generator.load_weights(args.load_generator_path)

    logging.info("Generating music to %s ...", args.output_path)
    gan.generate_sample_to(args.output_path)
    logging.info("Song generated. Enjoy!")


if __name__ == '__main__':
    main()
    exit(0)
