#!/usr/bin/env sh
set -e

FILE_NAME=speech_commands_v0.01.tar.gz
URL=http://download.tensorflow.org/data/$FILE_NAME
DATASET_FOLDER=datasets/speech_commands

FILE="datasets/speech_commands/$FILE_NAME"

if [ ! -f "$FILE" ];then
echo "downloading $URL...\n"
wget -O $FILE $URL
fi

echo "extracting $FILE_NAME..."
TEMP_DIRECTORY=$DATASET_FOLDER/audio
mkdir -p $TEMP_DIRECTORY
tar -xzf datasets/speech_commands/$FILE_NAME -C $TEMP_DIRECTORY

echo "splitting the dataset into train, validation and test sets..."
python $DATASET_FOLDER/split_dataset.py $DATASET_FOLDER

rsync -av --remove-source-files datasets/speech_commands/valid/ datasets/speech_commands/train/ && rm -r datasets/speech_commands/valid

echo "done"
