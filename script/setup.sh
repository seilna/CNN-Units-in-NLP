#!/bin/sh

python download.py `id` setup.zip
unzip setup.zip
rm setup.zip

mv setup/bytenet_data ../code/models/bytenet/Data
mv setup/bytenet_pretrained_models ../code/models/bytenet/pretrained_models

mv setup/vdcnn_data ../code/models/vdcnn/data
mv setup/vdcnn_pretrained_models ../code/models/vdcnn/pretrained_models

rm -rf setup
