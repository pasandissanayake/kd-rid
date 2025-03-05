#!/bin/bash

conda activate pytorch
export CUDA_VISIBLE_DEVICES=1
echo $CUDA_VISIBLE_DEVICES

cd "/home/pasand/pid-reps/pid-and-reps/experiment4_5000spc"
python cifar10.py
python compute_measures.py