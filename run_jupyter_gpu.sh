#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
var=$(whoami)
echo $var
python test_cuda.py
jupyter notebook \
--port 6006 \
--no-browser