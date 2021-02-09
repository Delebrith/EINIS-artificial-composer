# EINIS-artificial-composer
Project implemented as a part of EINIS course at WUT

## How to run training

In order to train a new model, use ```train.py``` file. You can configure number
of epochs, dataset, sampling frequency, number of pretraining epochs, batch size
and choose GAN architecture with flags. 

As for the architecture there are 3 options available: simple_cnn (DCGAN with one-hot encoding
for notes in MIDI files), sequence_lstm (generator made of dense layers and discriminator containing
2 LSTM layers) and piano_roll_cnn (DCGAN processing RGB images representing pianoroll format
for many channels in MIDI file).

Example: 

```python train.py --gan_type lstm_gan --epochs 5000 --sampling_rate 50```

## How to run generation

In order to generate a piece of music, use ```generate.py``` file. You must configure
architecture type and pass path to models weights (.h5 file) and to output file.

Example:

```--gan_type piano_roll_cnn --output_path ../samples/outpianoroll.midi --load_generator_path ../models/2021_01_31_22_56_40_generator_pianoroll-DC-GAN_epoch_5900.hdf5```