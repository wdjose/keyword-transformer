import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
import numpy as np
import pickle
from tqdm import tqdm

# Data generation and augmentation code from: https://github.com/google-research/google-research/tree/master/kws_streaming
# License: https://github.com/google-research/google-research/blob/master/LICENSE
from kws_streaming.models import model_flags
from kws_streaming.models import model_params
from kws_streaming.data import input_data

# Use only CPU (no GPU)
import tensorflow.compat.v1 as tf1
tf1.config.set_visible_devices([], 'GPU')
visible_devices = tf1.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'

class Datagen():
    def __init__(self, batch_size=512, version=1, preprocess="raw"):

        # Set PATH to data sets (for example to speech commands V2):
        # They can be downloaded from
        # https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz
        # https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
        # https://docs.google.com/uc?export=download&id=1OAN3h4uffi5HS7eb7goklWeI2XPm1jCS
        # Files should be downloaded then extracted in the google-speech-commands directory
        dataset = "google-speech-commands"
        DATA_PATH = os.path.join("data", dataset, "data{}".format(version))

        FLAGS = model_params.Params()
        FLAGS.data_dir = DATA_PATH
        FLAGS.verbosity = logging.ERROR

        # set wanted words for V2_35 dataset
        if version == 3:
            FLAGS.wanted_words = 'visual,wow,learn,backward,dog,two,left,happy,nine,go,up,bed,stop,one,zero,tree,seven,on,four,bird,right,eight,no,six,forward,house,marvin,sheila,five,off,three,down,cat,follow,yes'
            FLAGS.split_data = 0

        # set speech feature extractor properties
        FLAGS.mel_upper_edge_hertz = 7600
        FLAGS.window_size_ms = 30.0
        FLAGS.window_stride_ms = 10.0
        FLAGS.mel_num_bins = 80
        FLAGS.dct_num_features = 40
        FLAGS.feature_type = 'mfcc_tf'
        FLAGS.preprocess = preprocess

        # for numerical correctness of streaming and non streaming models set it to 1
        # but for real use case streaming set it to 0
        FLAGS.causal_data_frame_padding = 0

        FLAGS.use_tf_fft = True
        FLAGS.mel_non_zero_only = not FLAGS.use_tf_fft

        # data augmentation parameters
        FLAGS.resample = 0.15
        FLAGS.time_shift_ms = 100
        FLAGS.use_spec_augment = 1
        FLAGS.time_masks_number = 2
        FLAGS.time_mask_max_size = 25
        FLAGS.frequency_masks_number = 2
        FLAGS.frequency_mask_max_size = 7
        FLAGS.pick_deterministically = 1

        self.flags = model_flags.update_flags(FLAGS)
        import absl
        absl.logging.set_verbosity(self.flags.verbosity)


        self.flags.batch_size = batch_size
        self.time_shift_samples = int((self.flags.time_shift_ms * self.flags.sample_rate) / 1000)


        tf1.disable_eager_execution()
        config = tf1.ConfigProto(device_count={'GPU': 0})
        self.sess = tf1.Session(config=config)
        # tf1.keras.backend.set_session(self.sess)

        self.audio_processor = input_data.AudioProcessor(self.flags)

    def dataLen(self, part):
        if part == "val":
            set_size = self.audio_processor.set_size('validation')
            return set_size // self.flags.batch_size
        elif part == "test":
            set_size = self.audio_processor.set_size('testing')
            return set_size

    def getData(self, part, offset):
        flags = self.flags
        audio_processor = self.audio_processor
        # Generate training chunks
        if part == "train":
            train_fingerprints, train_ground_truth = audio_processor.get_data(
                flags.batch_size, offset * flags.batch_size, flags, flags.background_frequency,
                flags.background_volume, self.time_shift_samples, 'training',
                flags.resample, flags.volume_resample, self.sess)
            return(train_fingerprints.astype(np.float32), train_ground_truth.astype(int))

        # Generate validation
        elif part == "val":
            validation_fingerprints, validation_ground_truth = audio_processor.get_data(
                flags.batch_size, offset, flags, 0.0, 0.0, 0, 'validation', 0.0, 0.0, self.sess)
            return (validation_fingerprints.astype(np.float32), validation_ground_truth.astype(int))

        # Generate testing
        elif part == "test":
            inference_batch_size = 1
            test_fingerprints, test_ground_truth = audio_processor.get_data(
                inference_batch_size, offset, flags, 0.0, 0.0, 0, 'testing', 0.0, 0.0, self.sess)
            return (test_fingerprints.astype(np.float32), test_ground_truth.astype(int))
