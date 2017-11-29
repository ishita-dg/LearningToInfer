import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import generative
import models

torch.manual_seed(1)

N_epoch = 10
    
# Urn example
N_blocks = 15
N_trials = 20
N_balls = 9
testN_blocks = int(N_blocks/5)
valN_blocks = int(N_blocks/5)
    

data = {}
data["inf_p"] = {}
data["uninf_p"] = {}
approx_models = {}
hier_rational_models = {}
block_rational_models = {}

var_fac = 10
mean_fac = 0.25
NUM_LABELS = 2
INPUT_SIZE = 3
nhid = 5

for cond in data:
    data[cond]["train"] = {}
    data[cond]["val"] = {}
    data[cond]["test"] = {}
    approx_models[cond] = models.MLPClassifier(NUM_LABELS, INPUT_SIZE, nhid)
    hier_rational_models[cond] = models.hier_model()
    block_rational_models[cond] = models.block_model()
    
    # Generate data
    
    if (cond == "uninf_p"):
        
        ps, ls = generative.assign_PL(N_balls, N_blocks, var_fac, 1.0, mean_fac)
        data[cond]["train"]["X"],  data[cond]["train"]["y"] = \
            generative.data_gen(ps, ls, N_trials, N_blocks, N_balls)
        
        ps, ls = generative.assign_PL(N_balls, valN_blocks, var_fac, 1.0, mean_fac)
        data[cond]["val"]["X"],  data[cond]["val"]["y"] = \
            generative.data_gen(ps, ls, N_trials, valN_blocks, N_balls)        
        
        ps, ls = generative.assign_PL(N_balls, testN_blocks, var_fac, mean_fac, 1.0)
        data[cond]["test"]["X"],  data[cond]["test"]["y"] = \
            generative.data_gen(ps, ls, N_trials, testN_blocks, N_balls)
    else:
        
        ps, ls = generative.assign_PL(N_balls, N_blocks, var_fac, mean_fac, 1.0)
        data[cond]["train"]["X"],  data[cond]["train"]["y"] = \
            generative.data_gen(ps, ls, N_trials, N_blocks, N_balls)
        
        ps, ls = generative.assign_PL(N_balls, valN_blocks, var_fac, mean_fac, 1.0)
        data[cond]["val"]["X"],  data[cond]["val"]["y"] = \
            generative.data_gen(ps, ls, N_trials, valN_blocks, N_balls)

        ps, ls = generative.assign_PL(N_balls, testN_blocks, var_fac, 1.0, mean_fac)
        data[cond]["test"]["X"],  data[cond]["test"]["y"] = \
            generative.data_gen(ps, ls, N_trials, testN_blocks, N_balls)
        
    
        
    # train models
    # approx
    optimizer = optim.SGD(approx_models[cond].parameters(), lr=0.1)
    approx_models[cond].train(data[cond], N_epoch, optimizer)
            
    
for cond in data:
    
    print("Condition is : ", cond)
    print("Val")
    approx_models[cond].test(data[cond]["val"])
    print("Test")
    approx_models[cond].test(data[cond]["test"])
    


def plot_both(dset):
    
    count = 0
    fig = plt.figure(figsize=(16,8))
    for cond in data:
        count += 1
        ax = fig.add_subplot(1, 2, count)
        ax.plot(-2*data[cond][dset]["X"].numpy()[:, 0] + 1, label = "likl/ball_drawn")
        ax.plot(data[cond][dset]["X"].numpy()[:, 2], label = "pri/N_left")
        ax.plot(data[cond][dset]["y_pred"] - 0.5, label = "prediction")
        ax.plot(0.5 - data[cond][dset]["y"].numpy(), label = "true", linestyle = "--")
        ax.set_title('{0}'.format(cond))
        ax.set_ylim([-1.4, 1.4])
        ax.legend()
        
        plt.savefig('{0}_m{1}_v{2}epochs{3}.png'.format(dset,round(100.0/(1.0 + mean_fac)), var_fac, N_epoch))
        
        
plot_both("val")
plot_both("test")
    