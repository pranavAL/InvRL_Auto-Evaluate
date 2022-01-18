#!/bin/bash
python inference.py -mp 2seq_lstm_vae_Forced_.pth -sess 15 -ts True &&
python inference.py -mp 4seq_lstm_vae_Forced_.pth -sess 15  -ts True &&
python inference.py -mp 8seq_lstm_vae_Forced_.pth -sess 15  -ts True &&
python inference.py -mp 16seq_lstm_vae_Forced_.pth -sess 15  -ts True &&
python inference.py -mp 32seq_lstm_vae_Forced_.pth -sess 15 -ts True 
