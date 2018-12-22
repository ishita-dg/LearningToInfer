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
  total_part = 20
  
YV_all_hrms = []
YV_all_ams = []
YV_all_priors = []
YV_all_liks = []

NV_all_hrms = []
NV_all_ams = []
NV_all_priors = []
NV_all_liks = []

for part_number in np.arange(total_part):
  print("Participant number, ", part_number)
  # Modify in the future to read in / sysarg
  config = {'N_part' : part_number,
            'optimization_params': {'train_epoch': 500,
                                   'test_epoch': 0,
                                   'L2': 0.0,
                                   'train_lr': 0.05,
                                   'test_lr' : 0.0},
            'network_params': {'NHID': 1,
                               'NONLIN' : 'rbf'}}
  
  
  expt = generative.Urn()
  expt_name = "FC" 
  config['expt_name'] = expt_name
  
  # Parameters for generating the training data
  
  N_trials = 1
  
  train_blocks = 2
  test_blocks = 200
  N_blocks = train_blocks + test_blocks
  
  N_balls = 100
  
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
  
  # Varied data vs uninformative data
  
  YV_approx_model = expt.get_approxmodel(OUT_DIM, INPUT_SIZE, NHID, NONLIN)
  YV_rational_model = expt.get_rationalmodel(N_trials) 
  YV_block_vals =  expt.assign_PL_FC(N_balls, train_blocks, True)
  YV_X = expt.data_gen(YV_block_vals, N_trials, fixed = True)
  
  
  NV_approx_model = expt.get_approxmodel(OUT_DIM, INPUT_SIZE, NHID, NONLIN)
  NV_rational_model = expt.get_rationalmodel(N_trials) 
  NV_block_vals =  expt.assign_PL_FC(N_balls, train_blocks, False)
  NV_X = expt.data_gen(NV_block_vals, N_trials, fixed = True)
  
  
  test_block_vals = expt.assign_PL_FC(N_balls, train_blocks, True)
  test_X = expt.data_gen(test_block_vals, N_trials, fixed = True)
  
  # Create the data frames
  YV_train_data = {'X': YV_X,
                   'log_joint': None,
                   'y_hrm': None,
                   'y_am': None,
                   }
  
  
  YV_test_data = {'X': test_X,
                  'y_hrm': None,
                  'y_am': None,
                  }
  
  NV_train_data = {'X': NV_X,
                   'log_joint': None,
                   'y_hrm': None,
                   'y_am': None,
                   }
  
  NV_test_data = {'X': test_X,
                  
                  'y_hrm': None,
                  'y_am': None,
                  }
  
  # training models
  YV_train_data = YV_rational_model.train(YV_train_data)
  YV_approx_model.optimizer = optim.SGD(YV_approx_model.parameters(), 
                                        lr=train_lr, 
                                        weight_decay = L2)
  YV_approx_model.train(YV_train_data, train_epoch)
  #utils.save_model(YV_approx_model, name = storage_id + 'YV_trained_model')
  
  
  NV_train_data = NV_rational_model.train(NV_train_data)
  NV_approx_model.optimizer = optim.SGD(NV_approx_model.parameters(), 
                                        lr=train_lr, 
                                        weight_decay = L2)
  NV_approx_model.train(NV_train_data, train_epoch)
  #utils.save_model(NV_approx_model, name = storage_id + 'NV_trained_model')
  
  # testing models
  YV_test_data = YV_rational_model.test(YV_test_data)
  YV_approx_model.optimizer = optim.SGD(YV_approx_model.parameters(), 
                                        lr=test_lr)
  YV_test_data = YV_approx_model.test(YV_test_data, test_epoch, N_trials)
  #utils.save_model(YV_approx_model, name = storage_id + 'YV_tested_model')
  
  for key in YV_test_data:
    if type(YV_test_data[key]) is torch.FloatTensor:
      YV_test_data[key] = YV_test_data[key].numpy()
    else:
      YV_test_data[key] = np.array(YV_test_data[key])
      
  #utils.save_data(YV_test_data, name = storage_id + 'YV_test_data')
  
  
  NV_test_data = NV_rational_model.test(NV_test_data)
  NV_approx_model.optimizer = optim.SGD(NV_approx_model.parameters(), 
                                        lr=test_lr)
  NV_test_data = NV_approx_model.test(NV_test_data, test_epoch, N_trials)
  #utils.save_model(NV_approx_model, name = storage_id + 'NV_tested_model')
  
  for key in NV_test_data:
    if type(NV_test_data[key]) is torch.FloatTensor:
      NV_test_data[key] = NV_test_data[key].numpy()
    else:
      NV_test_data[key] = np.array(NV_test_data[key])
      
  #utils.save_data(NV_test_data, name = storage_id + 'NV_test_data')
  
  YV_all_hrms.append(YV_test_data['y_hrm'][:, 1])
  YV_all_ams.append(YV_test_data['y_am'][:, 1])
  YV_all_priors.append(YV_test_data['X'][:, -2])
  lik_of_data = (YV_test_data['X'][:, 0]*YV_test_data['X'][:, 2]
                 + (1.0 - YV_test_data['X'][:, 0])*(1.0 - YV_test_data['X'][:, 2]))
  YV_all_liks.append(lik_of_data)
  
  NV_all_hrms.append(NV_test_data['y_hrm'][:, 1])
  NV_all_ams.append(NV_test_data['y_am'][:, 1])
  NV_all_priors.append(NV_test_data['X'][:, -2])
  lik_of_data = (NV_test_data['X'][:, 0]*NV_test_data['X'][:, 2]
                 + (1.0 - NV_test_data['X'][:, 0])*(1.0 - NV_test_data['X'][:, 2]))
  NV_all_liks.append(lik_of_data)

    
  
  
YV_all_hrms = np.reshape(np.array(YV_all_hrms), (-1))
YV_all_ams = np.reshape(np.array(YV_all_ams), (-1))
YV_all_priors = np.reshape(np.array(YV_all_priors), (-1))
YV_all_liks = np.reshape(np.array(YV_all_liks), (-1))

NV_all_hrms = np.reshape(np.array(NV_all_hrms), (-1))
NV_all_ams = np.reshape(np.array(NV_all_ams), (-1))
NV_all_priors = np.reshape(np.array(NV_all_priors), (-1))
NV_all_liks = np.reshape(np.array(NV_all_liks), (-1))

prior_low = YV_all_priors < 0.5
prior_high = YV_all_priors > 0.5

# Plotting
fig, ax = plt.subplots(1, 1)

low_YV = np.mean(YV_all_ams[prior_low])
high_YV = np.mean(YV_all_ams[prior_high])

low_NV = np.mean(NV_all_ams[prior_low])
high_NV = np.mean(NV_all_ams[prior_high])

low_opt = np.mean(YV_all_hrms[prior_low])
high_opt = np.mean(YV_all_hrms[prior_high])

ax.bar([0, 1, 2], 
       [np.abs(high_opt - low_opt), np.abs(high_YV - low_YV), np.abs(high_NV - low_NV)])
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Optimal', 'Within-subject', 'Between-subject'])
ax.set_ylabel('Reaction to prior')

plt.show()
plt.savefig('figs/baserate_bars' + storage_id + '.pdf')

        

plot_data = {'YV_priors': YV_all_priors, 
             'YV_liks': YV_all_liks, 
             'YV_ams': YV_all_ams, 
             'YV_hrms': YV_all_hrms,              
             'NV_priors': NV_all_priors, 
             'NV_liks': NV_all_liks, 
             'NV_ams': NV_all_ams, 
             'NV_hrms': NV_all_hrms}

utils.save_data(plot_data, name = storage_id + 'plot_data')
        
        
