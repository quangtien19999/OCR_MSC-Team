#!/bin/bash

python raw_detect.py
python post_detect.py
python predict_merge.py
python post_merge.py