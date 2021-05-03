# KWS Transformer

This repository contains PyTorch code replicating the paper [Keyword Transformer: A Self-Attention Model for Keyword Spotting](https://arxiv.org/abs/2104.00769). Currently, KWT-1, KWT-2, and KWT-3 without distillation, with 40x1 input patches from the spectrogram. Distillation token (from MHAtt-RNN), support for other sizes of input patches, and support for V2-12 dataset evaluation to follow. 

Current model performance on v1 and v2 datasets (tweaks and improvements on-going): 

|       | V1-12 | V2-35 |
|-------|-------|-------|
| KWT-3 | 96.6% | 97.1% |
| KWT-2 | 96.3% | 97.0% |
| KWT-1 | 95.9% | 95.9% |

## Setup

Clone the repository: 
```bash
git clone https://github.com/wdjose/kws-transformer.git
cd kws-transformer
```

Create the google-speech-commands folder and download and extract the google-speech-commands dataset:
```bash
mkdir data
mkdir data/google-speech-commands
cd data/google-speech-commands
mkdir data1 data2
wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz
wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar -xzf speech_commands_v0.01.tar.gz -C data1
tar -xzf speech_commands_v0.02.tar.gz -C data2
rm speech_commands_v0.01.tar.gz speech_commands_v0.02.tar.gz
cd ../..
```

Clone the `kws_streaming` subdirectory in the google-research repository: 
```bash
svn export https://github.com/google-research/google-research/trunk/kws_streaming
```

## Replicate Original Keyword Transformer Models

Run data augmentation generation script (this runs using TensorFlow and pre-generates all 12M augmented MFCC samples for v1 and v2 datasets): 
```bash
bash datagen.sh
```

Train kws1, kws2, and kws3 variants on v1 and v2 datasets (no distillation): 
```bash
bash train_mfcc.sh
```
