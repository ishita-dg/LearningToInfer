import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import generative
import utils
import scipy.stats as ss

import matplotlib.pyplot as plt

import sys
import json



if len(sys.argv) > 1:
  total_part = int(sys.argv[1])
else:
  total_part = 20
  
atyp_hrms = []
atyp_ams = []
typ_hrms = []
typ_ams = []

for part_number in np.arange(total_part):
  print("Participant number, ", part_number)
  
  # Modify in the future to read in / sysarg
  config = {'N_part' : part_number,
            'optimization_params': {'train_epoch': 50,
                                   'test_epoch': 0,
                                   'L2': 0.0,
                                   'train_lr': 0.05,
                                   'test_lr' : 0.0},
            'network_params': {'NHID': 1,
                               'NONLIN' : 'rbf'},
            'N_balls' : 20,
            'train_blocks' : 200,
            'N_trials' : 1,
            'fac': 0.33}
  
  
  fac = np.random.choice([config['fac'], 1.0/config['fac']], 1)
  
  expt = generative.Urn()
  expt_name = "CS" # PM, PE, SR, EU, CP
  config['expt_name'] = expt_name
  
  # Parameters for generating the training data
  
  N_trials = config['N_trials']
  
  train_blocks = config['train_blocks']
  test_blocks = 10
  
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
  
  train_block_vals =  expt.assign_PL_CS(train_blocks + test_blocks, N_balls, alpha_post = fac, alpha_prior = 1.0)
  train_X = expt.data_gen(train_block_vals, N_trials, N_balls)
  test_block_vals =  expt.assign_PL_CS(test_blocks, N_balls, alpha_post = 1.0/fac, alpha_prior = 1.0)
  test_X = expt.data_gen(test_block_vals, N_trials)
  
  # Create the data frames
  train_data = {'X': train_X[:train_blocks*N_trials],
                'log_joint': None,
                'y_hrm': None,
                'y_am': None,
                }
  
  
  test_data = {'X': torch.stack((test_X, 
                                 train_X[-test_blocks*N_trials:])).view(-1, INPUT_SIZE),
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
  
  atyp_hrms.append(test_data['y_hrm'][:test_blocks*N_trials, 1])
  atyp_ams.append(test_data['y_am'][:test_blocks*N_trials, 1])
  typ_hrms.append(test_data['y_hrm'][-test_blocks*N_trials:, 1])
  typ_ams.append(test_data['y_am'][-test_blocks*N_trials:, 1])
  
  
typ_ams = np.reshape(np.array(typ_ams), (-1))
typ_hrms = np.reshape(np.array(typ_hrms), (-1))

atyp_ams = np.reshape(np.array(atyp_ams), (-1))
atyp_hrms = np.reshape(np.array(atyp_hrms), (-1))

notnan = np.logical_not(np.isnan(atyp_ams))
atyp_ams = atyp_ams[notnan]
atyp_hrms = atyp_hrms[notnan]


notnan = np.logical_not(np.isnan(typ_ams))
typ_ams = typ_ams[notnan]
typ_hrms = typ_hrms[notnan]

#which_urn = np.random.binomial(1, 0.5, ams.shape)
#ams = ams*which_urn + (1 - which_urn)*(1.0 - ams)
#hrms = hrms*which_urn + (1 - which_urn)*(1.0 - hrms)

# Plotting
f, (ax1, ax2) = plt.subplots(1, 2)

ax1.scatter(typ_hrms, typ_ams, label = 'Typical', s = 10)
slope, intercept, r_value, p_value, std_err = ss.linregress(typ_hrms, typ_ams)
ax1.plot(typ_hrms, typ_hrms*slope + intercept, c = 'k')
ax1.plot([0.0, 1.0], [0.0, 1.0], c = 'k', linestyle = ':')
ax1.set_title("Typical")
print "r-squared:", r_value**2


ax2.scatter(atyp_hrms, atyp_ams, label = 'Atypical', s = 10)
slope, intercept, r_value, p_value, std_err = ss.linregress(atyp_hrms, atyp_ams)
ax2.plot(atyp_hrms, atyp_hrms*slope + intercept, c = 'k')
ax2.plot([0.0, 1.0], [0.0, 1.0], c = 'k', linestyle = ':')
ax2.set_title("Atypical")
print "r-squared:", r_value**2

plt.show()
plt.savefig('figs/Precision_' + storage_id +'.pdf')


plot_data = {'typ_am': typ_ams,
             'atyp_am': atyp_ams,
             'typ_hrm': typ_hrms,
             'atyp_hrm': atyp_hrms}
utils.save_data(plot_data, name = storage_id + 'plot_data')