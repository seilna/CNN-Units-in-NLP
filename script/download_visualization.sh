#!/bin/sh

python download.py 136RtLeKwQHvhxH04rLt_ZiVImeDUdvpV visualization.zip
unzip visualization.zip
rm visualization.zip

mv visualization ../
