#!/bin/bash

# Generate parts 1 to 60 for google-speech-commands dataset v1 and v2
# Also generates validation and test sets
# Only mfcc samples are generated
# Total augmented samples take up ~350GB

for i in {1..60}
do
    python datagen.py $i --version 1
done
python datagen.py val --version 1
python datagen.py test --version 1

for i in {1..60}
do
    python datagen.py $i --version 2
done
python datagen.py val --version 2
python datagen.py test --version 2
