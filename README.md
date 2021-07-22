# Keyword Transformer

This repository contains PyTorch code replicating the paper [Keyword Transformer: A Self-Attention Model for Keyword Spotting](https://arxiv.org/abs/2104.00769). Currently, KWT-1, KWT-2, and KWT-3 without distillation, with 40x1 input patches from the spectrogram, are supported. Distillation token (from MHAtt-RNN) and support for other sizes of input patches are to follow. 

Replicated model performance on v1 and v2 datasets vs literature data: 

<table>
    <thead>
        <tr><th></th><th colspan=2>V1-12 Accuracy</th><th colspan=2>V2-12 Accuracy</th><th colspan=2>V2-35 Accuracy</th><th colspan=2># Parameters</th></tr>
        <tr><th></th><th>Replicated</th><th>Paper</th><th>Replicated</th><th>Paper</th><th>Replicated</th><th>Paper</th><th>Replicated</th><th>Paper</th></tr>
    </thead>
    <tbody>
        <tr><td>KWT-3</td><td>95.94%</td><td>97.24%</td><td>97.40%</td><td>98.54%</td><td>95.72%</td><td>97.51%</td><td>5,361k</td><td>5,361k</td></tr>
        <tr><td>KWT-2</td><td>95.46%</td><td>97.36%</td><td>97.08%</td><td>98.21%</td><td>95.85%</td><td>97.53%</td><td>2,394k</td><td>2,394k</td></tr>
        <tr><td>KWT-1</td><td>95.03%</td><td>97.05%</td><td>95.99%</td><td>97.72%</td><td>94.75%</td><td>96.85%</td><td>607k</td><td>607k</td></tr>
    </tbody>
</table>

## Setup

Clone the repository: 
```bash
git clone https://github.com/wdjose/keyword-transformer.git
cd keyword-transformer
```

Create the google-speech-commands folder and download and extract the google-speech-commands dataset:
```bash
mkdir -p data/google-speech-commands
cd data/google-speech-commands
mkdir -p data1 data2 data3
wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz
wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies 'https://docs.google.com/uc?export=download&id=1OAN3h4uffi5HS7eb7goklWeI2XPm1jCS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1OAN3h4uffi5HS7eb7goklWeI2XPm1jCS" -O data_all_v2.zip && rm -rf /tmp/cookies.txt
tar -xzf speech_commands_v0.01.tar.gz -C data1
tar -xzf speech_commands_v0.02.tar.gz -C data2
unzip data_all_v2.zip -d data3
rm speech_commands_v0.01.tar.gz speech_commands_v0.02.tar.gz data_all_v2.zip
cd ../..
```

Clone the `kws_streaming` subdirectory in the google-research repository: 
```bash
svn export https://github.com/google-research/google-research/trunk/kws_streaming
```

## Replicate Original Keyword Transformer Models

Train kwt1, kwt2, and kwt3 variants on v1 and v2 datasets with no distillation (this generates 12M augmented MFCC samples with TensorFlow for v1 and v2 datasets):
```bash
bash train.sh
```
For the purposes of this repository, version=1 (data1) corresponds to v1-12 (12 labels), version=2 (data2) corresponds to v2-12 (12 labels), and version=3 (data3) corresponds to v2-35 (35 labels). So technically, "version=3" refers to the modified v2 dataset (with 35 labels) as defined by the paper [Streaming Keyword Spotting on Mobile Devices](https://arxiv.org/abs/2005.06720), from which the data augmentation code came (the `kws_streaming` repository exported from above). 
