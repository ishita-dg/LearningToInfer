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

# Run results for reanalysis of Sam's experiment (SR)

expt = generative.Button()
expt_name = "SR" # PM, PE, SR, EU, CP
config['expt_name'] = expt_name

# Parameters for generating the training data

N_trials = 10

train_blocks = 10
test_blocks = 10
N_blocks = train_blocks + test_blocks

# Optimization parameters
train_epoch = config['optimization_params']['train_epoch']
test_epoch = config['optimization_params']['test_epoch']
L2 = config['optimization_params']['L2']
train_lr = config['optimization_params']['train_lr']
test_lr = config['optimization_params']['test_lr']

# Network parameters -- single hidden layer MLP
# Can also adjust the nonlinearity
OUT_DIM = 2 #mu and sigma
INPUT_SIZE = 3 #obs, mean_so_far, N in block
NHID = config['network_params']['NHID']

storage_id = utils.make_id(config)

# Informative data vs uninformative data

prior_variance = 144.0
lik_variance = 25.0

ID_approx_model = expt.get_approxmodel(OUT_DIM, INPUT_SIZE, NHID)
ID_rational_model = expt.get_rationalmodel(prior_variance, lik_variance, N_trials) 
ID_block_vals =  expt.assign_PL(N_blocks, prior_variance)
ID_X = expt.data_gen(ID_block_vals, lik_variance, N_trials)

prior_variance = 36.0
lik_variance = 25.0

UD_approx_model = expt.get_approxmodel(OUT_DIM, INPUT_SIZE, NHID)
UD_rational_model = expt.get_rationalmodel(prior_variance, lik_variance, N_trials) 
UD_block_vals =  expt.assign_PL(N_blocks, prior_variance)
UD_X = expt.data_gen(UD_block_vals, lik_variance, N_trials)

# Create the data frames
ID_train_data = {'X': ID_X[:train_blocks*N_trials],
                 'log_joint': None,
                 'y_hrm': None,
                 'y_am': None,
                 }


ID_test_data = {'X': ID_X[-test_blocks*N_trials:],
                'y_hrm': None,
                'y_am': None,
                }

UD_train_data = {'X': UD_X[:train_blocks*N_trials],
                 'log_joint': None,
                 'y_hrm': None,
                 'y_am': None,
                 }

UD_test_data = {'X': UD_X[-test_blocks*N_trials:],
                'y_hrm': None,
                'y_am': None,
                }

# training models
ID_train_data = ID_rational_model.train(ID_train_data)
ID_approx_model.optimizer = optim.SGD(ID_approx_model.parameters(), 
                                      lr=train_lr, 
                                      weight_decay = L2)
ID_approx_model.train(ID_train_data, train_epoch)
utils.save_model(ID_approx_model, name = storage_id + 'ID_trained_model')


UD_train_data = UD_rational_model.train(UD_train_data)
UD_approx_model.optimizer = optim.SGD(UD_approx_model.parameters(), 
                                      lr=train_lr, 
                                      weight_decay = L2)
UD_approx_model.train(UD_train_data, train_epoch)
utils.save_model(UD_approx_model, name = storage_id + 'UD_trained_model')

# testing models
ID_test_data = ID_rational_model.test(ID_test_data)
ID_approx_model.optimizer = optim.SGD(ID_approx_model.parameters(), 
                                      lr=test_lr)
ID_test_data = ID_approx_model.test(ID_test_data, test_epoch, N_trials)
utils.save_model(ID_approx_model, name = storage_id + 'ID_tested_model')
utils.save_data(ID_test_data, name = storage_id + 'ID_test_data')

UD_test_data = UD_rational_model.test(UD_test_data)
UD_approx_model.optimizer = optim.SGD(UD_approx_model.parameters(), 
                                      lr=test_lr)
UD_test_data = UD_approx_model.test(UD_test_data, test_epoch, N_trials)
utils.save_model(UD_approx_model, name = storage_id + 'UD_tested_model')
utils.save_data(UD_test_data, name = storage_id + 'UD_test_data')

        
        
        