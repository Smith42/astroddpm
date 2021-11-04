#!/bin/bash

SORTED=$(sort -k10 -t, SDSS_DR7_noheader.csv)

echo $SORTED | xargs -n1024 | parallel --jobs 8 --delimiter ' ' ./bother_desi {}

python ./preprocess.py
