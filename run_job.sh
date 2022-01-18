#!/bin/bash
python train_LSTM.py -sq 2 -mt Forced &&
python train_LSTM.py -sq 4 -mt Forced &&
python train_LSTM.py -sq 8 -mt Forced &&
python train_LSTM.py -sq 16 -mt Forced &&
python train_LSTM.py -sq 32 -mt Forced
