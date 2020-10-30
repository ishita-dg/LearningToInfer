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
  
ID_all_hrms = []
ID_all_ams = []

UD_all_hrms = []
UD_all_ams = []

ID_Xs = []
UD_Xs = []


for part_number in np.arange(total_part):
  print("Participant numpber ", part_number)
  # Modify in the future to read in / sysarg
  config = {'N_part' : part_number,
            'optimization_params': {'train_epoch': 30,
                                   'test_epoch': 0,
                                   'L2': 0.0,
                                   'train_lr': 0.02,
                                   'test_lr' : 0.0},
            'network_params': {'NHID': 2,
                               'NONLIN' : 'tanh'}}
  
  # Run model for reanalysis of Sam's experiment (SR)
  
  expt = generative.Button()
  expt_name = "SR" # PM, PE, SR, EU, CP
  config['expt_name'] = expt_name
  
  # Parameters for generating the training data
  
  N_trials = 10
  
  train_blocks = 10
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
  OUT_DIM = 2 #mu and sigma
  INPUT_SIZE = 3 #obs, mean_so_far, N in block
  NHID = config['network_params']['NHID']
  NONLIN = config['network_params']['NONLIN']
  
  storage_id = utils.make_id(config)
  
  # Informative data vs uninformative data
  
  prior_variance = 144.0
  lik_variance = 25.0
  
  ID_approx_model = expt.get_approxmodel(OUT_DIM, INPUT_SIZE, NHID, NONLIN)
  ID_rational_model = expt.get_rationalmodel(prior_variance, lik_variance, N_trials) 
  ID_block_vals =  expt.assign_PL(N_blocks, prior_variance)
  ID_X = expt.data_gen(ID_block_vals, lik_variance, N_trials)
  
  prior_variance = 36.0
  lik_variance = 25.0
  
  UD_approx_model = expt.get_approxmodel(OUT_DIM, INPUT_SIZE, NHID, NONLIN)
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
  #utils.save_model(ID_approx_model, name = storage_id + 'ID_trained_model')
  
  
  UD_train_data = UD_rational_model.train(UD_train_data)
  UD_approx_model.optimizer = optim.SGD(UD_approx_model.parameters(), 
                                        lr=train_lr, 
                                        weight_decay = L2)
  UD_approx_model.train(UD_train_data, train_epoch)
  #utils.save_model(UD_approx_model, name = storage_id + 'UD_trained_model')
  
  # testing models
  ID_test_data = ID_rational_model.test(ID_test_data)
  ID_approx_model.optimizer = optim.SGD(ID_approx_model.parameters(), 
                                        lr=test_lr)
  ID_test_data = ID_approx_model.test(ID_test_data, test_epoch, N_trials)
  #utils.save_model(ID_approx_model, name = storage_id + 'ID_tested_model')
  
  for key in ID_test_data:
    if type(ID_test_data[key]) is torch.FloatTensor:
      ID_test_data[key] = ID_test_data[key].numpy()
    else:
      ID_test_data[key] = np.array(ID_test_data[key])
        
  #utils.save_data(ID_test_data, name = storage_id + 'ID_test_data')
  
  UD_test_data = UD_rational_model.test(UD_test_data)
  UD_approx_model.optimizer = optim.SGD(UD_approx_model.parameters(), 
                                        lr=test_lr)
  UD_test_data = UD_approx_model.test(UD_test_data, test_epoch, N_trials)
  #utils.save_model(UD_approx_model, name = storage_id + 'UD_tested_model')
  
  for key in UD_test_data:
    if type(UD_test_data[key]) is torch.FloatTensor:
      UD_test_data[key] = UD_test_data[key].numpy()
    else:
      UD_test_data[key] = np.array(UD_test_data[key])
  
  
  #utils.save_data(UD_test_data, name = storage_id + 'UD_test_data')
  ID_all_hrms.append(ID_test_data['y_hrm'][:, 0])
  ID_all_ams.append(ID_test_data['y_am'][:, 0])
  
  UD_all_hrms.append(UD_test_data['y_hrm'][:, 0])
  UD_all_ams.append(UD_test_data['y_am'][:, 0])
  
  ID_Xs.append(ID_test_data['X'])
  UD_Xs.append(ID_test_data['X'])
  
    
    
ID_all_hrms = np.reshape(np.array(ID_all_hrms), (-1))
ID_all_ams = np.reshape(np.array(ID_all_ams), (-1))

UD_all_hrms = np.reshape(np.array(UD_all_hrms), (-1))
UD_all_ams = np.reshape(np.array(UD_all_ams), (-1))

ID_Xs = np.reshape(np.array(ID_Xs), (-1, 3))
UD_Xs = np.reshape(np.array(UD_Xs), (-1, 3))

   
#print("Plotting now")  
  
#ID_all_hrms0 = ID_all_hrms.copy()
#UD_all_hrms0 = UD_all_hrms.copy()

#ID_all_ams0 = ID_all_ams.copy()
#UD_all_ams0 = UD_all_ams.copy()

#ID_all_ams = ID_all_ams0 - ID_Xs[:, 1]
#UD_all_ams = UD_all_ams0 - UD_Xs[:, 1]

#ID_all_hrms = ID_all_hrms0 - ID_Xs[:, 1]
#UD_all_hrms = UD_all_hrms0 - UD_Xs[:, 1]

    
#ID_all_ams = ID_all_ams0[1:] - ID_all_ams0[:-1]
#UD_all_ams = UD_all_ams0[1:] - UD_all_ams0[:-1]

#ID_all_hrms = ID_all_hrms0[1:] - ID_all_hrms0[:-1]
#UD_all_hrms = UD_all_hrms0[1:] - UD_all_hrms0[:-1]


# Plotting
f, ax = plt.subplots(1, 1)
jump = 2.0
ID_means = []
UD_means = []

bins = np.arange(0, 18.0, jump)
digitized = np.digitize(np.abs(ID_all_hrms), bins)
for d in np.arange(len(bins)):
  ID_means.append(np.mean(np.abs(ID_all_ams)[digitized == d+1]))
ax.plot(bins+jump/2.0,ID_means, label = 'Inf Data')

bins = np.arange(0, 13.0, jump)
digitized = np.digitize(np.abs(UD_all_hrms), bins)
for d in np.arange(len(bins)):
  UD_means.append(np.mean(np.abs(UD_all_ams)[digitized == d+1]))
ax.plot(bins+jump/2.0,UD_means, label = 'Uninf Data')

ax.plot([0, 20], [0, 20], c = 'k')

plt.legend()
plt.show()
plt.savefig('figs/Updates_' + storage_id +'.pdf')

## Plotting
#f, ax = plt.subplots(1, 1)
#jump = 2.0
#ID_means = []
#UD_means = []

#a_mu = lambda x: np.mean(np.abs(x))
#a_er = lambda x: 1.96 * np.std(np.abs(x)) / (np.sqrt(len(x)))
#ticks = [0, 0.4, 1, 1.4]

#ax.bar(ticks, 
       #[a_mu(ID_all_ams), a_mu(ID_all_hrms), a_mu(UD_all_ams), a_mu(UD_all_hrms)], 
       #yerr = [a_er(ID_all_ams), a_er(ID_all_hrms), a_er(UD_all_ams), a_er(UD_all_hrms)], width = 0.15)
#ax.set_xticks(ticks)
#ax.set_xticklabels(['ID model', 'ID true', 'UD model', 'UD true'])
#ax.set_ylabel('Update')


#plt.legend()
#plt.show()
#plt.savefig('figs/updates_bar' + storage_id + '.pdf')

    

plot_data = {'ID_ams': ID_all_ams,
             'UD_ams': UD_all_ams,
             'ID_hrms': ID_all_hrms,
             'UD_hrms': UD_all_hrms,
             'x': bins + jump/2.0,
             'UD_means': np.array(UD_means),
             'ID_means': np.array(ID_means),
             'ID_X': ID_Xs,
             'UD_X': UD_Xs
             }

utils.save_data(plot_data, name = storage_id + 'plot_data')


