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
  total_part = 1

hrms = []
ams = []
all_priors = []
conds = []
Ns = []
liks = []

for part_number in np.arange(total_part):
  
  print("Participant number, ", part_number + 1)
  
  # Modify in the future to read in / sysarg
  config = {'N_part' : part_number,
            'fix_prior': False,
            'fix_ll': False,
            'optimization_params': {'train_epoch': int(sys.argv[3]),
                                   'test_epoch': 0,
                                   'L2': 0.0,
                                   'train_lr': 0.02,
                                   'test_lr' : 0.0},
           'network_params': {'NHID': int(sys.argv[2]),
                              'NONLIN' : 'rbf'}}
  
  
  # Run results for reanalysis of Philip and Edwards (PE)
  
  expt = generative.Urn()
  expt_name = "PE" # PM, PE, SR, EU, CP
  config['expt_name'] = expt_name
  
  # Parameters for generating the training data
  
  N_trials = 20
  
  train_blocks = 15
  test_blocks = 300
  N_blocks = train_blocks + test_blocks
  
  N_balls = 100
  fix_prior = config['fix_prior']
  fix_ll = config['fix_ll']
  
  
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
  block_vals =  expt.assign_PL_replications(N_balls, N_blocks, expt_name, fix_prior, fix_ll)
  indices = np.repeat(block_vals[-1], N_trials)
  priors = np.repeat(block_vals[0], N_trials)  
  X, true_urns = expt.data_gen(block_vals[:-1], N_trials, same_urn = True, return_urns = True)
  
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
  all_priors.append(test_data['prior'])
  conds.append(test_data['l_cond'])
  keep = (true_urns[-test_blocks*N_trials:] * (test_data['X'][:,2] > test_data['X'][:,1]) +
          (1 - true_urns[-test_blocks*N_trials:]) * (test_data['X'][:,2] < test_data['X'][:,1]))
  corrected_N = ((1 - keep)*test_data['X'][:,-1] - 
                 (keep)*test_data['X'][:,-1])
  Ns.append(corrected_N)
  lik = test_data['X'][:, 1]/ test_data['X'][:, 2]
  sample = test_data['X'][:, 0]
  liks.append(lik*sample + (1.0 - sample)/lik)
  
  
ams = np.reshape(np.array(ams), (-1))
hrms = np.reshape(np.array(hrms, dtype = 'float64'), (-1))
all_priors = np.reshape(np.array(all_priors), (-1))
conds = np.reshape(np.array(conds), (-1))
Ns = np.reshape(np.array(Ns), (-1))
liks = np.reshape(np.array(liks), (-1))

notnan = np.logical_not(np.isnan(ams))


clip_mask, _, ARs = utils.find_AR(hrms, ams, 1.0 - all_priors, randomize = False, clip = [-0.0, 100])

#ARs = ARs[clip_mask]
#conds = conds[clip_mask]
#Ns = Ns[clip_mask]


plot_data = {'Ns': Ns[notnan],
             'conds': conds[notnan],
             'priors': all_priors[notnan],
             'ams': ams[notnan],
             'hrms': hrms[notnan],
             'liks': liks[notnan]}


## Plotting
#fig, ax = plt.subplots(1, 1)
#for cond in np.sort(np.unique(conds)):
  #mask = conds == cond
  #x, y = Ns[mask], ARs[mask]
  #Y_means = []
  #Y_errs = []
  #Xs = []
  #ps = np.sort(np.unique(x))
  #for p in ps:
    #if sum(x == p) > len(x)/30.0:
      #Y_means.append(np.mean(y[x == p]))
      #Y_errs.append(np.std(y[x == p]))
      #Xs.append(p*20)
  #ax.plot(Xs, Y_means, label = str(cond))
  #ax.set_xlim([-6, 18])
  #ax.set_ylim([0.0, 3])
  #ax.axhline(1.0, c = 'k')
  #ax.axvline(0.0, c = 'k')
  #plot_data['x' + str(cond)] = np.array(Xs)
  #plot_data['y' + str(cond)] = np.array(Y_means)
  ##ax.scatter(x,y, label = str(cond))
  
#plt.legend()
#plt.show()
#plt.savefig('figs/AR_' + storage_id + 'full_0cutoff.pdf')

utils.save_data(plot_data, name = storage_id + 'plot_data')      
