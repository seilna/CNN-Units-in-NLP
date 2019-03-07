#!/bin/sh

python download.py `id` visualization.zip
unzip visualization.zip
rm visualization.zip

mv visualization ../
