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



if len(sys.argv) > 1:
  total_part = int(sys.argv[1])
else:
  total_part = 50

hrms = []
ams = []
priors = []
lik_ratios = []
strengths = []
weights = []

for part_number in np.arange(total_part):
  
  print("Participant number, ", part_number + 1)
  
  # Modify in the future to read in / sysarg
  config = {'N_part' : part_number,
            'Bias' : False, 
            'optimization_params': {'train_epoch': 30,
                                   'test_epoch': 0,
                                   'L2': 0.0,
                                   'train_lr': 0.01,
                                   'test_lr' : 0.0},
           'network_params': {'NHID': 1,
                              'NONLIN' : 'rbf'}}
  
  
  # Run results for reanalysis of Philip and Edwards (PE)
  
  expt = generative.Urn()
  expt_name = "Benj" # PM, PE, SR, EU, CP
  config['expt_name'] = expt_name
  
  # Parameters for generating the training data
  
  N_trials = 1
  sample_sizes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 33, 48])
  
  train_blocks = 150
  test_blocks = 300
  N_blocks = train_blocks + test_blocks
  
  # Optimization parameters
  train_epoch = config['optimization_params']['train_epoch']
  test_epoch = config['optimization_params']['test_epoch']
  L2 = config['optimization_params']['L2']
  train_lr = config['optimization_params']['train_lr']
  test_lr = config['optimization_params']['test_lr']
  
  # Network parameters -- single hidden layer MLP
  # Can also adjust the nonlinearity
  OUT_DIM = 2
  INPUT_SIZE = 6 #data, lik1, lik2, prior, strength, weight
  NHID = config['network_params']['NHID']
  NONLIN = config['network_params']['NONLIN']
  
  storage_id = utils.make_id(config)
  
  # Informative data vs uninformative data
  
  approx_model = expt.get_approxmodel(OUT_DIM, INPUT_SIZE, NHID, NONLIN)
  rational_model = expt.get_rationalmodel(N_trials) 
  block_vals =  expt.assign_PL_demo(N_blocks, bias = config['Bias'])
  ss_vals = np.random.choice(sample_sizes, N_blocks)
  X, true_urns = expt.data_gen(block_vals[:-1], N_trials, same_urn = True, return_urns = True, variable_ss = ss_vals)
  
  # Create the data frames
  train_data = {'X': X[train_blocks*N_trials:],
                'log_joint': None,
                'y_hrm': None,
                'y_am': None,
                }
  
  
  test_data = {'X': X[-test_blocks*N_trials:],
               'y_hrm': None,
               'y_am': None,
               }
  
  # Convert Xs to GT decomposition
  
  
  # training models
  train_data = rational_model.train_GT(train_data, max_N = max(ss_vals))
  approx_model.optimizer = optim.SGD(approx_model.parameters(), 
                                        lr=train_lr, 
                                        weight_decay = L2)
  approx_model.train(train_data, train_epoch)
  #utils.save_model(approx_model, name = storage_id + 'trained_model')
  
  # testing models
  test_data = rational_model.test_GT(test_data, max_N = max(ss_vals))
  approx_model.optimizer = optim.SGD(approx_model.parameters(), 
                                        lr=test_lr)
  test_data = approx_model.test(test_data, test_epoch, N_trials)
  #test_data['ARs'] = utils.find_AR(test_data['y_hrm'], test_data['y_am'], test_data['prior'])
  #utils.save_model(approx_model, name = storage_id + 'tested_model')
  
  
  for key in test_data:
    if type(test_data[key]) is torch.FloatTensor:
      test_data[key] = test_data[key].numpy()
    else:
      test_data[key] = np.array(test_data[key])
      
  #utils.save_data(test_data, name = storage_id + 'test_data')
  
  hrms.append(test_data['y_hrm'][:, 1])
  ams.append(test_data['y_am'][:, 1])
  strengths.append(test_data['X'][:, -2])
  weights.append(test_data['X'][:, -1])
  priors.append(test_data['X'][:, 3])
  lik_ratios.append(test_data['X'][:, 1] / test_data['X'][:, 2])
  
  

ams = np.reshape(np.array(ams), (-1))
hrms = np.reshape(np.array(hrms), (-1))
strengths = np.reshape(np.array(strengths), (-1))
weights = np.reshape(np.array(weights), (-1))
priors = np.reshape(np.array(priors), (-1))
lik_ratios = np.reshape(np.array(lik_ratios), (-1))


notnan = np.logical_not(np.isnan(ams))

plot_data = {'ams': ams[notnan],
             'hrms': hrms[notnan],
             'strengths': strengths[notnan],
             'weights': weights[notnan],
             'priors': priors[notnan],
             'lik_ratios': lik_ratios[notnan]}


utils.save_data(plot_data, name = storage_id + 'plot_data')      
# Plotting

