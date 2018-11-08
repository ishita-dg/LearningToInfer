import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy 

import generative
import utils

import matplotlib.pyplot as plt

import sys
import json


#*************************************

def KLs(model, objects, cond_dists):
         preds = []
         KL = []
         for o, true in zip(objects, cond_dists):
                  pred = model(autograd.Variable(o)).data.numpy()
                  pred = np.exp(pred)
                  pred /= sum(pred)
                  preds.append(pred)
                  KL.append(utils.find_KL(pred, true) + utils.find_KL(true,pred))
         
         return(np.array(preds), KL)

#*************************************


# Modify in the future to read in / sysarg
config = {'N_part' : 0,
          'optimization_params': {'train_epoch': 30,
                                 'test_epoch': 5,
                                 'L2': 0.0,
                                 'train_lr': 0.05,
                                 'test_lr' : 0.01},
          'network_params': {'NHID': 10}}

# Run results for old sample-based amortization experiment (OA)

expt = generative.Urn()
expt_name = "OA"
config['expt_name'] = expt_name

# Parameters for generating the training data

N_objects = 12
N_dim = N_objects # one-hot for now
alpha = 0.1
eta = 0.1

#N_trials = 1

#train_blocks = 100
#test_blocks = 10
#N_blocks = train_blocks + test_blocks

#N_balls = 10

# Optimization parameters
train_epoch = config['optimization_params']['train_epoch']
test_epoch = config['optimization_params']['test_epoch']
L2 = config['optimization_params']['L2']
train_lr = config['optimization_params']['train_lr']
test_lr = config['optimization_params']['test_lr']

# Network parameters -- single hidden layer MLP
# Can also adjust the nonlinearity
OUT_DIM = N_objects
INPUT_SIZE = N_dim #data, lik1, lik2, prior, N
NHID = config['network_params']['NHID']

storage_id = utils.make_id(config)

# Generate data
objects = np.eye(N_objects)
objects = torch.from_numpy(objects)
objects = objects.type(torch.FloatTensor)

# Choose 2 topics that are very different looking
ts = np.random.dirichlet(eta*np.ones(N_objects), 30)
max = 0
topic1 = topic2 = ts[0]
for i, t1 in enumerate(ts):
         for t2 in ts[i:]:
                  dist = (utils.find_KL(t1, t2)+
                          utils.find_KL(t2, t1))
                  if dist > max:
                           max = dist
                           topic1 = t1
                           topic2 = t2

high_po_indicator = topic1+topic2 > np.median(topic1+topic2)
N_hp_objects = np.sum(high_po_indicator)

# Derive the conditional distributions from these topics
# Approximate for now with no mixing of topics, alpha = 0.0

#Matrix is j | i
conditionals = np.empty((N_objects, N_objects))
log_joint = np.empty((N_objects, N_objects))
for i in np.arange(N_objects):
         norm = topic1[i] + topic2[i]
         conditionals[i, :] = (topic1*topic1[i] + topic2*topic2[i]) / norm
         log_joint[i, :] = np.log(topic1*topic1[i] + topic2*topic2[i])
 
log_joint = torch.from_numpy(log_joint)
log_joint = log_joint.type(torch.FloatTensor)
train_data = {'X': objects,
              'log_joint': log_joint}

# Make approximate model
approx_model = expt.get_approxmodel(OUT_DIM, INPUT_SIZE, NHID)


KLmat = np.zeros((N_hp_objects, N_hp_objects))
for i in np.arange(N_hp_objects):
         for j in np.arange(N_hp_objects):
                  KLmat[i,j] = utils.find_KL(conditionals[high_po_indicator][i, ], conditionals[high_po_indicator][j, ])


fig, ax = plt.subplots()
im = ax.imshow(KLmat, cmap=plt.get_cmap('hot'), interpolation='nearest',
               vmin=0, vmax=5)
fig.colorbar(im)
plt.savefig('figs/KLheatmap')

print("Median KL = ", np.median(KLmat[~np.eye(N_hp_objects, dtype=bool)]))

#training models
approx_model.optimizer = optim.SGD(approx_model.parameters(), 
                                   lr=train_lr, 
                                   weight_decay = L2)
approx_model.train(train_data, train_epoch)
utils.save_model(approx_model, name = storage_id + 'trained_model')

# Small stochastic updates when answering Q1

subadditive = np.empty((N_hp_objects, N_hp_objects))
superadditive = np.empty((N_hp_objects, N_hp_objects))

approx_model.optimizer = optim.SGD(approx_model.parameters(), 
                           lr=0.0)
tested_data = approx_model.test(train_data, 1, 1)
base = tested_data['y_am'].numpy()[np.outer(high_po_indicator, high_po_indicator)]
base = base.reshape((N_hp_objects, -1))

for i, co in enumerate(np.arange(N_objects)[high_po_indicator]):
         for j, qo in enumerate(np.arange(N_objects)[high_po_indicator]):
                  if (co != qo):                           
                           
                           sub_log_joint = copy.deepcopy(log_joint)
                           sub_log_joint[co, qo] += np.log(5)
                           
                           sub_train_data = {'X': objects[co, :].view(1, -1),
                                             'log_joint': sub_log_joint[co, :].view(1, -1)}
                           
                           model = copy.deepcopy(approx_model)
                           model.optimizer = optim.SGD(model.parameters(), 
                                                       lr=test_lr)
                           model.train(sub_train_data, test_epoch)
                           
                           tested_data = model.test(train_data, 1, 1)
                           subadditive[i, j] = tested_data['y_am'].numpy()[co, qo]
                           
                           #**************************
                           
                           super_log_joint = copy.deepcopy(log_joint)
                           super_log_joint[co, qo] -= np.log(5)
                           
                           super_train_data = {'X': objects[co, :].view(1, -1),
                                             'log_joint': super_log_joint[co, :].view(1, -1)}
                           
                           model = copy.deepcopy(approx_model)
                           model.optimizer = optim.SGD(model.parameters(), 
                                                       lr=test_lr)
                           model.train(super_train_data, test_epoch)
                           
                           tested_data = model.test(train_data, 1, 1)
                           superadditive[i, j] = tested_data['y_am'].numpy()[co, qo]
                           

K = 5
sub = 1.0 - (1.0 - subadditive[~np.eye(N_hp_objects, dtype=bool)])**K
super = 1.0 - (1.0 - superadditive[~np.eye(N_hp_objects, dtype=bool)])**K
control = 1.0 - (1.0 - base[~np.eye(N_hp_objects, dtype=bool)])**K
KLs = KLmat[~np.eye(N_hp_objects, dtype=bool)]
highKL_vals = 100*np.vstack(((sub-control)[KLs > np.median(KLs)], 
                         (super-control)[KLs > np.median(KLs)]))
lowKL_vals = 100*np.vstack(((sub-control)[KLs < np.median(KLs)], 
                         (super-control)[KLs < np.median(KLs)]))

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.bar([0, 1], np.mean(lowKL_vals, axis = 1), 
        yerr = 1.96*np.std(lowKL_vals, axis = 1)/np.sqrt(lowKL_vals.shape[1]))
ax1.set_title('Low KL')
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['Subadditivity', 'Superadditivity'])
ax2.bar([0,1], np.mean(highKL_vals, axis = 1), 
        yerr = 1.96*np.std(highKL_vals, axis = 1)/np.sqrt(highKL_vals.shape[1]))
ax2.set_title('High KL')
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['Subadditivity', 'Superadditivity'])
plt.show()
plt.savefig('figs/Compare_' + storage_id + 'K' + str(K) + '.pdf')

         