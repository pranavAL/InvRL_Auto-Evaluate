#!/bin/bash
python inference.py -mp 2seq_lstm_vae.pth &&
python inference.py -mp 4seq_lstm_vae.pth &&
python inference.py -mp 8seq_lstm_vae.pth &&
python inference.py -mp 16seq_lstm_vae.pth &&
python inference.py -mp 32seq_lstm_vae.pth
