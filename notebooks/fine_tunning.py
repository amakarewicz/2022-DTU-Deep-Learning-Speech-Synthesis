# importing diffwave functions
from diffwave import dataset
from diffwave.params import params
from diffwave.model import *
from diffwave.learner import *
from diffwave import preprocess
import logging

# and some libraries
import numpy as np
import argparse

try:
    args = argparse.Namespace()
    args.dir = '../data/LJSpeech-1.1'
    spectrogram = preprocess.main(args)
except Exception as Argument:
    logging.exception('Error in creating spectrograms.')

try:
    dataloader = dataset.from_path(['../data/LJSpeech-1.1/wav'], params)
    model = DiffWave(params)
    optimizer = torch.optim.Adam(model.parameters())
    Learner = DiffWaveLearner(model_dir = '../model/conditional_model', model = model, dataset = dataloader, optimizer = optimizer, params = params)
except Exception as Argument:
    logging.exception('Error in creating Learner.')

try:
    Learner.load_state_dict(torch.load('diffwave-ljspeech-22kHz-1000578.pt'))
except Exception as Argument:
    logging.exception('Error in loading pre-trained model.')

Learner.model = nn.Sequential(*list(Learner.model.children())[:4]) + nn.Sequential(*list(Learner.model.children())[5:])
# Learner.train
