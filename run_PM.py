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



hrms = []
ams = []
all_priors = []
conds = []

for part_number in np.arange(20):
  print("Participant number, ", part_number)
  
  # Modify in the future to read in / sysarg
  config = {'N_part' : part_number,
            'optimization_params': {'train_epoch': 30,
                                   'test_epoch': 0,
                                   'L2': 0.0,
                                   'train_lr': 0.05,
                                   'test_lr' : 0.0},
            'network_params': {'NHID': 2}}
  
  # Run results for reanalysis of Peterson and Miller (PM)
  
  expt = generative.Urn()
  expt_name = "PM" # PM, PE, SR, EU, CP
  config['expt_name'] = expt_name
  
  # Parameters for generating the training data
  
  N_trials = 1
  
  train_blocks = 200
  test_blocks = 300
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
  block_vals =  expt.assign_PL_replications(N_balls, N_blocks, expt_name)
  indices = np.repeat(block_vals[-1], N_trials)
  priors = np.repeat(block_vals[0], N_trials) 
  X = expt.data_gen(block_vals[:-1], N_trials)
  
  # Create the data frames
  train_data = {'X': X[:train_blocks*N_trials],
                'l_cond' : indices[:train_blocks*N_trials],
                'prior' : priors[:train_blocks*N_trials],
                'log_joint': None,
                'y_hrm': None,
                'y_am': None,
                }
  
  
  test_data = {'X': X[-test_blocks*N_trials:],
               'l_cond' : indices[-test_blocks*N_trials:],
               'prior' : priors[-test_blocks*N_trials:],
               'y_hrm': None,
               'y_am': None,
               }
  
  
  # training models
  train_data = rational_model.train(train_data)
  approx_model.optimizer = optim.SGD(approx_model.parameters(), 
                                        lr=train_lr, 
                                        weight_decay = L2)
  approx_model.train(train_data, train_epoch)
  #utils.save_model(approx_model, name = storage_id + 'trained_model')
  
  # testing models
  test_data = rational_model.test(test_data)
  approx_model.optimizer = optim.SGD(approx_model.parameters(), 
                                        lr=test_lr)
  test_data = approx_model.test(test_data, test_epoch, N_trials)
  #utils.save_model(approx_model, name = storage_id + 'tested_model')
  
  
  for key in test_data:
    if type(test_data[key]) is torch.FloatTensor:
      test_data[key] = test_data[key].numpy()
    else:
      test_data[key] = np.array(test_data[key])
      
  #utils.save_data(test_data, name = storage_id + 'test_data')
  hrms.append(test_data['y_hrm'][:, 1])
  ams.append(test_data['y_am'][:, 1])
  all_priors.append(test_data['prior'])
  conds.append(test_data['l_cond'])
  
ams = np.reshape(np.array(ams), (-1))
hrms = np.reshape(np.array(hrms), (-1))
all_priors = np.reshape(np.array(all_priors), (-1))
all_conds = np.reshape(np.array(conds), (-1))
  
  
# Plotting
clip_mask, priors, all_ARs = utils.find_AR(hrms, ams, all_priors, randomize = True, clip = [-00.0, 100])

priors = priors[clip_mask]
ARs = all_ARs[clip_mask]
conds = all_conds[clip_mask]

#which_urn = np.random.binomial(1, 1.0, test_data['prior'].shape)
#new_priors = which_urn*test_data['prior'] + (1 - which_urn)*(1.0-test_data['prior'])

fig, ax = plt.subplots(1, 1)
jump = 0.1
bins = np.arange(0.0, 1.0, jump)

for cond in np.sort(np.unique(conds)):
  mask = conds == cond
  x, y = 1.0 - priors[mask], ARs[mask]
  Y_means = []
  Y_errs = []
  digitized = np.digitize(x, bins)
  for d in np.arange(len(bins)):
    Y_means.append(np.mean(y[digitized == d+1]))
    Y_errs.append(np.var(y[digitized == d+1]))
  
  ax.plot(bins, Y_means, label = str(cond))
  ax.axhline(1.0, c = 'k')
  #ax.scatter(x,y, label = str(cond))
  
plt.legend()
plt.show()
plt.savefig('figs/AR_' + storage_id + 'full_0cutoff.pdf')

        
