# KWS Transformer

This repository contains code replicating the paper [Keyword Transformer: A Self-Attention Model for Keyword Spotting](https://arxiv.org/abs/2104.00769). 

Further experiments on different model architectures and improvements are also included in this repository. 

## Setup

Clone the repository: 
```bash
git clone https://github.com/wdjose/kws-transformer.git
cd kws-transformer
```

Create the google-speech-commands folder and download and extract the google-speech-commands dataset:
```bash
mkdir google-speech-commands
cd google-speech-commands
mkdir data1 data2
wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz
wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar -xzf speech_commands_v0.01.tar.gz -C data1
tar -xzf speech_commands_v0.02.tar.gz -C data2
rm speech_commands_v0.01.tar.gz speech_commands_v0.02.tar.gz
cd ..
```

Clone the `kws_streaming` subdirectory in the google-research folder: 
```bash
svn export https://github.com/google-research/google-research/trunk/kws_streaming
```

## Replicate Original Keyword Transformer Models

Run data augmentation generation script: 
```bash
bash datagen.sh
```

Train kws1, kws2, and kws3 variants on v1 and v2 datasets (no distillation): 
```bash
bash train_mfcc.sh
```
