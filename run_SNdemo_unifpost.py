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
  total_part = 5

hrms = []
ams = []

for part_number in np.arange(total_part):
  print("Participant number, ", part_number)
  
  # Modify in the future to read in / sysarg
  config = {'N_part' : part_number,
            'optimization_params': {'train_epoch': 100,
                                    'train_blocks': 200,
                                    'test_epoch': 0,
                                   'L2': 0.0,
                                   'train_lr': 0.05,
                                   'test_lr' : 0.0},
            'network_params': {'NHID': 10,
                               'NONLIN' : 'rbf'},
            'N_balls' : 100,
            'alpha_pre' : 1.0, 
            'N_trials' : 1}
  
  # Run results for Correction Prior (CP)
  
  expt = generative.Urn()
  expt_name = "SN" # PM, PE, SR, EU, CP
  config['expt_name'] = expt_name
  
  # Parameters for generating the training data
  
  N_trials = config['N_trials']
  
  train_blocks = config['optimization_params']['train_blocks']
  test_blocks = 500
  N_blocks = train_blocks + test_blocks
  
  N_balls = config['N_balls']
  
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
  NONLIN = config['network_params']['NONLIN']
  
  storage_id = utils.make_id(config)
  
  # Informative data vs uninformative data
  
  approx_model = expt.get_approxmodel(OUT_DIM, INPUT_SIZE, NHID, NONLIN)
  rational_model = expt.get_rationalmodel(N_trials) 
  
  train_block_vals =  expt.assign_PL_CP(train_blocks, N_balls, alpha_post = 1.0, alpha_pre = config['alpha_pre'])
  train_X = expt.data_gen(train_block_vals, N_trials, N_balls)
  test_block_vals =  expt.assign_PL_CP(test_blocks, N_balls, alpha_post = 1.0, alpha_pre = config['alpha_pre'])
  test_X = expt.data_gen(test_block_vals, N_trials)
  
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
  if True :
    hrms.append(test_data['y_hrm'][:, 1])
    ams.append(test_data['y_am'][:, 1])
  else:
    print("**********reject this participant")
  
  
ams = np.reshape(np.array(ams), (-1))
hrms = np.reshape(np.array(hrms), (-1))
which_urn = np.random.binomial(1, 0.5, ams.shape)
ams = ams*which_urn + (1 - which_urn)*(1.0 - ams)
hrms = hrms*which_urn + (1 - which_urn)*(1.0 - hrms)

notnan = np.logical_not(np.isnan(ams))
ams = ams[notnan]
hrms = hrms[notnan]


#Plotting 

fig, ax = plt.subplots(1, 1)
jump = 0.1
bins = np.arange(0.0, 1.0, jump)
plot_data = {
             'ams': ams,
             'hrms': hrms
             }


ax.scatter(hrms, ams)
ax.plot([0.0, 1.0],[0.0, 1.0], c = 'k')

plt.legend()
#plt.show()
plt.savefig('figs/Demo_' + storage_id + '.pdf')

utils.save_data(plot_data, name = storage_id + 'plot_data')
        