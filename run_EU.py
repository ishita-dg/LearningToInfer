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


  
ID_all_hrms = []
ID_all_ams = []
ID_all_priors = []

UD_all_hrms = []
UD_all_ams = []
UD_all_priors = []

for part_number in np.arange(6):
  # Modify in the future to read in / sysarg
  config = {'N_part' : part_number,
            'optimization_params': {'train_epoch': 50,
                                   'test_epoch': 0,
                                   'L2': 0.0,
                                   'train_lr': 0.05,
                                   'test_lr' : 0.0},
            'network_params': {'NHID': 2}}
  
  # Run results for Eric's Urn experiment (EU)
  
  expt = generative.Urn()
  expt_name = "EU" # PM, PE, SR, EU, CP
  config['expt_name'] = expt_name
  
  # Parameters for generating the training data
  
  N_trials = 1
  
  train_blocks = 100
  test_blocks = 200
  N_blocks = train_blocks + test_blocks
  
  N_balls = 10
  
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
  
  ID_approx_model = expt.get_approxmodel(OUT_DIM, INPUT_SIZE, NHID)
  ID_rational_model = expt.get_rationalmodel(N_trials) 
  ID_block_vals =  expt.assign_PL_EU(N_balls, N_blocks, True)
  ID_X = expt.data_gen(ID_block_vals, N_trials)
  
  
  UD_approx_model = expt.get_approxmodel(OUT_DIM, INPUT_SIZE, NHID)
  UD_rational_model = expt.get_rationalmodel(N_trials) 
  UD_block_vals =  expt.assign_PL_EU(N_balls, N_blocks, False)
  UD_X = expt.data_gen(UD_block_vals, N_trials)
  
  
  # Create the data frames
  ID_train_data = {'X': ID_X[:train_blocks*N_trials],
                   'log_joint': None,
                   'y_hrm': None,
                   'y_am': None,
                   }
  
  
  ID_test_data = {'X': torch.stack((UD_X[-test_blocks*N_trials:], 
                                    ID_X[-test_blocks*N_trials:])).view(-1, INPUT_SIZE),
                  'y_hrm': None,
                  'y_am': None,
                  }
  
  UD_train_data = {'X': UD_X[:train_blocks*N_trials],
                   'log_joint': None,
                   'y_hrm': None,
                   'y_am': None,
                   }
  
  UD_test_data = {'X': torch.stack((UD_X[-test_blocks*N_trials:], 
                                    ID_X[-test_blocks*N_trials:])).view(-1, INPUT_SIZE),
                  
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
  
  for key in ID_test_data:
    if type(ID_test_data[key]) is torch.FloatTensor:
      ID_test_data[key] = ID_test_data[key].numpy()
    else:
      ID_test_data[key] = np.array(ID_test_data[key])
      
  utils.save_data(ID_test_data, name = storage_id + 'ID_test_data')
  
  
  UD_test_data = UD_rational_model.test(UD_test_data)
  UD_approx_model.optimizer = optim.SGD(UD_approx_model.parameters(), 
                                        lr=test_lr)
  UD_test_data = UD_approx_model.test(UD_test_data, test_epoch, N_trials)
  utils.save_model(UD_approx_model, name = storage_id + 'UD_tested_model')
  
  for key in UD_test_data:
    if type(UD_test_data[key]) is torch.FloatTensor:
      UD_test_data[key] = UD_test_data[key].numpy()
    else:
      UD_test_data[key] = np.array(UD_test_data[key])
      
  utils.save_data(UD_test_data, name = storage_id + 'UD_test_data')
  
  ID_all_hrms.append(ID_test_data['y_hrm'][:, 1])
  ID_all_ams.append(ID_test_data['y_am'][:, 1])
  ID_all_priors.append(ID_test_data['X'][:, -2])
  
  UD_all_hrms.append(UD_test_data['y_hrm'][:, 1])
  UD_all_ams.append(UD_test_data['y_am'][:, 1])
  UD_all_priors.append(UD_test_data['X'][:, -2])
  
    
  
  
ID_all_hrms = np.reshape(np.array(ID_all_hrms), (-1))
ID_all_ams = np.reshape(np.array(ID_all_ams), (-1))
ID_all_priors = np.reshape(np.array(ID_all_priors), (-1))

UD_all_hrms = np.reshape(np.array(UD_all_hrms), (-1))
UD_all_ams = np.reshape(np.array(UD_all_ams), (-1))
UD_all_priors = np.reshape(np.array(UD_all_priors), (-1))

# Plotting
fig, ax = plt.subplots(1, 1)
priors, ID_ARs = utils.find_AR(ID_all_hrms, 
                       ID_all_ams, 
                       1.0 - ID_all_priors, 
                       randomize = False, clip = [-100.0, 100])

priors, UD_ARs = utils.find_AR(UD_all_hrms, 
                       UD_all_ams, 
                       1.0 - UD_all_priors, 
                       randomize = False, clip = [-100.0, 100])


#ID_Y_means = []
#UD_Y_means = []
#ps = np.sort(np.unique(priors))
#for p in ps:
  #ID_Y_means.append(np.mean(ID_ARs[priors == p]))
  #UD_Y_means.append(np.mean(UD_ARs[priors == p]))

#ax.plot(ps, ID_Y_means, label = 'Inf Data')
#ax.plot(ps, UD_Y_means, label = 'Uninf Data')
#ax.axhline(1.0, c = 'k')

ax.bar([0, 1], 
       [np.mean(ID_ARs), np.mean(UD_ARs)], 
       yerr = [1.96*np.std(ID_ARs)/(np.sqrt(len(ID_ARs))), 1.96*np.std(UD_ARs)/(np.sqrt(len(UD_ARs)))])
ax.set_xticks([0, 1])
ax.set_xticklabels(['Informative Data', 'Uninformative Data'])
ax.set_ylabel('AR on test')


plt.legend()
plt.show()
plt.savefig('figs/AR_bar_' + storage_id + '.pdf')

        

        
        
        
