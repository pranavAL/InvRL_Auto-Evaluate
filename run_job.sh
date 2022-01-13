#!/bin/bash
python train_LSTM.py -sq 2 &&
python train_LSTM.py -sq 4 &&
python train_LSTM.py -sq 8 &&
python train_LSTM.py -sq 16 &&
python train_LSTM.py -sq 32