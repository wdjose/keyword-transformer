import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
import argparse
import numpy as np
import tensorflow.compat.v1 as tf1
import pickle
from tqdm import tqdm
from kws_streaming.models import model_flags
from kws_streaming.models import model_params
from kws_streaming.data import input_data

parser = argparse.ArgumentParser()
parser.add_argument("part", help="Part number (1 to 60 if batch_size=512 and num_batches=384) or \"val\" or \"test\"")
parser.add_argument("--version", default=1, type=int, choices=[1, 2], help="Google speech commands version (1 or 2)")
parser.add_argument("--preprocess", default="mfcc", choices=["mfcc", "raw"], help="Type of generated samples (mfcc or raw)")
parser.add_argument("--batch-size", default=512, type=int, help="Number of samples per batch")
parser.add_argument("--num-batches", default=384, type=int, help="Number of batches per chunk")
args = parser.parse_args()

part = args.part
batch_size = args.batch_size
num_batches = args.num_batches
version = args.version
preprocess = args.preprocess

if part.isnumeric():
    if batch_size==512 and num_batches==384:
        print("Generating part {} of 60".format(part))
    else:
        print("Generating part {}".format(part))
elif part == "val" or part == "test":
    print("Generating {}".format(part))


# Set PATH to data sets (for example to speech commands V2):
# It can be downloaded from
# https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz
# https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
# Files should be downloaded then extracted in the google-speech-commands directory
dataset = "google-speech-commands"
DATA_PATH = os.path.join("data", dataset, "data{}".format(version))

if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists(os.path.join('data', 'v{}_{}'.format(version, preprocess))):
    os.makedirs(os.path.join('data', 'v{}_{}'.format(version, preprocess)))

FLAGS = model_params.Params()
FLAGS.data_dir = DATA_PATH
FLAGS.verbosity = logging.ERROR

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

flags = model_flags.update_flags(FLAGS)
import absl
absl.logging.set_verbosity(flags.verbosity)


flags.batch_size = batch_size
time_shift_samples = int((flags.time_shift_ms * flags.sample_rate) / 1000)


tf1.disable_eager_execution()
config = tf1.ConfigProto()
sess = tf1.Session(config=config)
tf1.keras.backend.set_session(sess)

audio_processor = input_data.AudioProcessor(flags)

# Generate training chunks
if part != "val" and part != "test":        
    samples_list = []
    training_step = 1
    for i in tqdm(range(num_batches), file=sys.stdout):
        offset = ((int(part) - 1)*384 + training_step - 1) * flags.batch_size
        training_step += 1
        train_fingerprints, train_ground_truth = audio_processor.get_data(
            flags.batch_size, offset, flags, flags.background_frequency,
            flags.background_volume, time_shift_samples, 'training',
            flags.resample, flags.volume_resample, sess)
        samples_list.append((train_fingerprints.astype(np.float32), 
            train_ground_truth.astype(int)))
    with open(os.path.join("data", "v{}_{}".format(version, preprocess), "{}.pkl".format(part)), 'wb') as f:
        pickle.dump(samples_list, f)

# Generate validation
elif part == "val":
    set_size = audio_processor.set_size('validation')
    samples_list = []
    training_step = 1
    for i in tqdm(range(0, set_size, flags.batch_size), file=sys.stdout):
        validation_fingerprints, validation_ground_truth = audio_processor.get_data(
            flags.batch_size, i, flags, 0.0, 0.0, 0, 'validation', 0.0, 0.0, sess)
        samples_list.append((validation_fingerprints.astype(np.float32), 
            validation_ground_truth.astype(int)))
    with open(os.path.join("data", "v{}_{}".format(version, preprocess), "val.pkl"), 'wb') as f:
        pickle.dump(samples_list, f)

# Generate testing
elif part == "test":
    set_size = audio_processor.set_size('testing')
    inference_batch_size = 1
    samples_list = []
    training_step = 1
    for i in tqdm(range(0, set_size, inference_batch_size), file=sys.stdout):
        test_fingerprints, test_ground_truth = audio_processor.get_data(
            inference_batch_size, i, flags, 0.0, 0.0, 0, 'testing', 0.0, 0.0, sess)
        samples_list.append((test_fingerprints.astype(np.float32), 
            test_ground_truth.astype(int)))
    with open(os.path.join("data", "v{}_{}".format(version, preprocess), "test.pkl"), 'wb') as f:
        pickle.dump(samples_list, f)

print("Done {}".format(part))
exit()
