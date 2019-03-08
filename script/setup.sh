#!/bin/sh

echo "Downloading pretrained models & training data..."
python download.py 1FzyRIJHrIEk480_WrLeXcfg58XJIFwOY download.zip
unzip download.zip
rm download.zip

mv download/bytenet_data ../code/models/bytenet/Data
mv download/bytenet_pretrained_models ../code/models/bytenet/pretrained_models

mv download/vdcnn_data ../code/models/vdcnn/data
mv download/vdcnn_pretrained_models ../code/models/vdcnn/pretrained_models

rm -rf setup
