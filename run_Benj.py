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
  nhid = int(sys.argv[2])
  train_e = int(sys.argv[3])
else:
  total_part = 15*10
  nhid = 2
  train_e = 50

hrms = []
ams = []
priors = []
lik_ratios = []
strengths = []
weights = []
which = []

for part_number in np.arange(total_part):
  
  print("Participant number, ", part_number + 1)
  
  # Modify in the future to read in / sysarg
  config = {'N_part' : part_number,
            'diff_noise': True,
            'noise_blocks' : 150, 
            'query_manip': True, 
            'train_blocks' : 150,
            'optimization_params': {'train_epoch': train_e,
                                   'test_epoch': 0,
                                   'L2': 0.0,
                                   'train_lr': 0.01,
                                   'test_lr' : 0.0},
           'network_params': {'NHID': nhid,
                              'NONLIN' : 'rbf'}}
  
  
  # Run results for reanalysis of Philip and Edwards (PE)
  
  expt = generative.Urn()
  expt_name = "Benj" # PM, PE, SR, EU, CP
  config['expt_name'] = expt_name
  
  sub_es = ['PM65', 'PSM65', 'GT92', 'BH80', 'DD74', 'BWB70', 
            'GHR65', 'Gr92', 'HS09', 'KW04', 'KW04', 'MC72', 'MC72', 
            'Ne01', 'SK07']
  E = sub_es[part_number%len(sub_es)]
    
  
  # Parameters for generating the training data
  
  N_trials = 1
  noise_blocks = config['noise_blocks']
  train_blocks = config['train_blocks']
  net_blocks = noise_blocks + train_blocks
  
  if ((E == 'PSM65' or E == 'GT92') and config['diff_noise']):
    noise_blocks = int(noise_blocks/2)
    train_blocks = net_blocks - noise_blocks
    
  test_blocks = 100
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
  X, max_N = expt.data_gen_Benj(N_blocks, which = E)
  
  # equalize prior for test cases
  #X[:, 5] = 0.5
  test_X = X[-test_blocks*N_trials:, :]
  test_X[:, 3] = 0.5
  #filter_prior = (test_X[:, 3] == 0.5).view((-1,1))
  #test_X = torch.masked_select(test_X, filter_prior).view((-1,6))
  

  max_r_ss = min(10, max_N)
  block_vals =  expt.assign_PL_demo(noise_blocks)
  random_ss = np.random.choice(1.0 + np.arange(max_r_ss), noise_blocks)
  random_X = expt.data_gen(block_vals[:-1], 1, variable_ss = random_ss)
  
  print(E)
  
  # query distribution manipulation
  train_X = torch.cat((random_X,
                       X[:train_blocks*N_trials]), dim = 0)
  
  if config['query_manip']:
    new_Ns = np.random.choice([0.0, 1.0/max_N, -1.0/max_N], 
                              net_blocks, 
                              p = [0.8, 0.1, 0.1])
    new_Ns = torch.from_numpy(new_Ns)
    new_Ns = new_Ns.type(torch.FloatTensor)    
    #train_X[:, 1] = 0.7
    #train_X[:, 2] = 0.3
    #train_X[:, 2] = 0.5
    train_X[:, 4] = new_Ns
    #train_X[:, 5] = 1.0
    #maxN = 8
  
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
  
  # Convert Xs to GT decomposition
  
  
  # training models
  train_data = rational_model.train_GT(train_data, max_N = max_N)
  approx_model.optimizer = optim.SGD(approx_model.parameters(), 
                                        lr=train_lr, 
                                        weight_decay = L2)
  approx_model.train(train_data, train_epoch)
  #utils.save_model(approx_model, name = storage_id + 'trained_model')
  
  # testing models
  test_data = rational_model.test_GT(test_data, max_N = max_N)
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
  weights.append(test_data['X'][:, -1]*max_N)
  priors.append(test_data['X'][:, 3])
  lik_ratios.append(test_data['X'][:, 1] / test_data['X'][:, 2])
  which.append([E]*test_blocks)
  
  

ams = np.reshape(np.array(ams), (-1))
hrms = np.reshape(np.array(hrms), (-1))
strengths = np.reshape(np.array(strengths), (-1))
weights = np.reshape(np.array(weights), (-1))
priors = np.reshape(np.array(priors), (-1))
lik_ratios = np.reshape(np.array(lik_ratios), (-1))

notnan = np.logical_not(np.isnan(ams))
which = list(np.reshape(np.array(which), (-1))[notnan])

plot_data = {'ams': ams[notnan],
             'hrms': hrms[notnan],
             'strengths': strengths[notnan],
             'weights': weights[notnan],
             'priors': priors[notnan],
             'lik_ratios': lik_ratios[notnan],
             'sub_exp': which}


utils.save_data(plot_data, name = storage_id + 'unif_samples_plot_data')      
# Plotting

