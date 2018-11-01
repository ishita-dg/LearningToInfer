import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import generative
import utils

import matplotlib.pyplot as plt

import sys
import json

# Modify in the future to read in / sysarg
config = {'N_part' : 0,
          'optimization_params': {'train_epoch': 10,
                                 'test_epoch': 0,
                                 'L2': 0.0,
                                 'train_lr': 0.05,
                                 'test_lr' : 0.0},
          'network_params': {'NHID': 2}}

# Run results for Correction Prior (CP)

expt = generative.Urn()
expt_name = "CP" # PM, PE, SR, EU, CP
config['expt_name'] = expt_name

# Parameters for generating the training data

N_trials = 1

train_blocks = 200
test_blocks = 200
N_blocks = train_blocks + test_blocks

N_balls = 6

# Optimization parameters
train_epoch = config['optimization_params']['train_epoch']
test_epoch = config['optimization_params']['test_epoch']
L2 = config['optimization_params']['L2']
train_lr = config['optimization_params']['train_lr']
test_lr = config['optimization_params']['test_lr']

# Network parameters -- single hidden layer MLP
# Can also adjust the nonlinearity
OUT_DIM = 2
INPUT_SIZE = 5 #data, lik1, lik2, prior, N
NHID = config['network_params']['NHID']

storage_id = utils.make_id(config)

# Informative data vs uninformative data

approx_model = expt.get_approxmodel(OUT_DIM, INPUT_SIZE, NHID)
rational_model = expt.get_rationalmodel(N_trials) 

train_block_vals =  expt.assign_PL_CP(train_blocks, N_balls, alpha = 0.27)
train_X = expt.data_gen(train_block_vals, N_trials, N_balls)
test_block_vals =  expt.assign_PL_CP(train_blocks, N_balls, alpha = 1.0)
test_X = expt.data_gen(test_block_vals, N_trials, N_balls)

# Create the data frames
train_data = {'X': train_X,
              'log_joint': None,
              'y_hrm': None,
              'y_am': None,
              }


test_data = {'X': test_X,
             'y_hrm': None,
             'y_am': None,
             }


# training models
train_data = rational_model.train(train_data)
approx_model.optimizer = optim.SGD(approx_model.parameters(), 
                                      lr=train_lr, 
                                      weight_decay = L2)
approx_model.train(train_data, train_epoch)
utils.save_model(approx_model, name = storage_id + 'trained_model')

# testing models
test_data = rational_model.test(test_data)
approx_model.optimizer = optim.SGD(approx_model.parameters(), 
                                      lr=test_lr)
test_data = approx_model.test(test_data, test_epoch, N_trials)
utils.save_model(approx_model, name = storage_id + 'tested_model')
utils.save_data(test_data, name = storage_id + 'test_data')

        
