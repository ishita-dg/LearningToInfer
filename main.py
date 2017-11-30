import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import generative
import utils

import matplotlib.pyplot as plt

torch.manual_seed(6)

N_epoch = 5
sg_epoch = 3
    
N_blocks = 10
N_trials = 60
N_balls = 9
testN_blocks = 1
valN_blocks = 3

expts = {"disc": {},
         "cont": {}}

expts["disc"]["expt_type"] = generative.Urn()
expts["cont"]["expt_type"] = generative.Button()

# running only for disc expt (Urn)   

for expt in ["disc"]:
    
    expts[expt]["data"] = {}
    expts[expt]["data"]["inf_p"] = {}
    expts[expt]["data"]["uninf_p"] = {}
    expts[expt]["am"] = {}
    expts[expt]["hrm"] = {}
    expts[expt]["block_rational_models"] = {}
    

    fac = 300.0
    prior_fac = 1
    NUM_LABELS = 2
    INPUT_SIZE = 4
    nhid = 5
    
    
    for cond in expts[expt]["data"]:
        expts[expt]["data"][cond]["train"] = {}
        expts[expt]["data"][cond]["val"] = {}
        expts[expt]["data"][cond]["test"] = {}
        expts[expt]["am"][cond] = expts[expt]["expt_type"].get_approxmodel(NUM_LABELS, INPUT_SIZE, nhid)
        expts[expt]["hrm"][cond] = expts[expt]["expt_type"].get_rationalmodel(prior_fac, N_trials)
        expts[expt]["block_rational_models"][cond] = expts[expt]["expt_type"].get_rationalmodel(prior_fac, N_trials)
        
        # Generate data
        
        if (cond == "uninf_p"):
            
            ps, ls = expts[expt]["expt_type"].assign_PL(N_balls, N_blocks, fac)
            expts[expt]["data"][cond]["train"]["X"],  expts[expt]["data"][cond]["train"]["y"] = \
                expts[expt]["expt_type"].data_gen(ps, ls, N_trials, N_blocks, N_balls)
            
            ps, ls = expts[expt]["expt_type"].assign_PL(N_balls, valN_blocks, fac)
            expts[expt]["data"][cond]["val"]["X"],  expts[expt]["data"][cond]["val"]["y"] = \
                expts[expt]["expt_type"].data_gen(ps, ls, N_trials, valN_blocks, N_balls)        
            
            ps, ls = expts[expt]["expt_type"].assign_PL(N_balls, testN_blocks, 1.0/fac)
            expts[expt]["data"][cond]["test"]["X"],  expts[expt]["data"][cond]["test"]["y"] = \
                expts[expt]["expt_type"].data_gen(ps, ls, N_trials, testN_blocks, N_balls)
        else:
            
            ps, ls = expts[expt]["expt_type"].assign_PL(N_balls, N_blocks, 1.0/fac)
            expts[expt]["data"][cond]["train"]["X"],  expts[expt]["data"][cond]["train"]["y"] = \
                expts[expt]["expt_type"].data_gen(ps, ls, N_trials, N_blocks, N_balls)
            
            ps, ls = expts[expt]["expt_type"].assign_PL(N_balls, valN_blocks, 1.0/fac)
            expts[expt]["data"][cond]["val"]["X"],  expts[expt]["data"][cond]["val"]["y"] = \
                expts[expt]["expt_type"].data_gen(ps, ls, N_trials, valN_blocks, N_balls)
    
            ps, ls = expts[expt]["expt_type"].assign_PL(N_balls, testN_blocks, fac)
            expts[expt]["data"][cond]["test"]["X"],  expts[expt]["data"][cond]["test"]["y"] = \
                expts[expt]["expt_type"].data_gen(ps, ls, N_trials, testN_blocks, N_balls)
            
        
            
        # train models
        # approx
        expts[expt]["am"][cond].optimizer = \
            optim.SGD(expts[expt]["am"][cond].parameters(), lr=0.1)
        expts[expt]["am"][cond].train(expts[expt]["data"][cond]["train"], N_epoch)
        # Block model doesn't get trained
        # Hierarchical model trains it's prior over mu_p and mu_l
        expts[expt]["hrm"][cond].train(expts[expt]["data"][cond]["train"])
                
        
    for cond in expts[expt]["data"]:
        
        print("Condition is : ", cond)
        print("Val")
        dset = "val"
        expts[expt]["am"][cond].test(expts[expt]["data"][cond][dset], sg_epoch, 'am')
        expts[expt]["hrm"][cond].test(expts[expt]["data"][cond][dset], 'hrm')
        print("Test")
        dset = "test"
        expts[expt]["am"][cond].test(expts[expt]["data"][cond][dset], sg_epoch, 'am')
        expts[expt]["hrm"][cond].test(expts[expt]["data"][cond][dset], 'hrm')
        
    
                
    
    utils.plot_both(expts[expt]["data"], 'hrm', "val", fac, N_epoch)
    utils.plot_both(expts[expt]["data"], 'hrm', "test", fac, N_epoch)
    utils.plot_both(expts[expt]["data"], 'am', "val", fac, N_epoch)
    utils.plot_both(expts[expt]["data"], 'am', "test", fac, N_epoch)
    
        
        
        
#parity = 2*expts[expt]["data"]["inf_p"]['test']["X"][1:-1,0].numpy() -1
parity = np.ones_like(expts[expt]["data"]["inf_p"]['test']["X"][1:-1,0].numpy())

prob = False

updates_inf = np.vstack((np.abs(utils.updates(expts[expt]["data"]["inf_p"]['test']["y_predhrm"], prob)), 
                         np.abs(utils.updates(expts[expt]["data"]["inf_p"]['test']["y_predam"], prob)),
                         np.abs(utils.updates(expts[expt]["data"]["inf_p"]['test']["X"][:,0].numpy())),
                         np.abs(utils.updates(expts[expt]["data"]["inf_p"]['test']["X"][:,2].numpy()))))

updates_uninf = np.vstack((np.abs(utils.updates(expts[expt]["data"]["uninf_p"]['test']["y_predhrm"], prob)), 
                         np.abs(utils.updates(expts[expt]["data"]["uninf_p"]['test']["y_predam"], prob)),
                         np.abs(utils.updates(expts[expt]["data"]["uninf_p"]['test']["X"][:,0].numpy())),
                         np.abs(utils.updates(expts[expt]["data"]["uninf_p"]['test']["X"][:,2].numpy()))))




#inf_hrm = np.cumsum(updates_inf[2,:]*updates_inf[0,:])
#uninf_hrm = np.cumsum(updates_uninf[2,:]*updates_uninf[0,:])

#inf_am = np.cumsum(updates_inf[2,:]*updates_inf[1,:])
#uninf_am = np.cumsum(updates_uninf[2,:]*updates_uninf[1,:])

inf_hrm = updates_inf[2,:]*updates_inf[0,:]
uninf_hrm = updates_uninf[2,:]*updates_uninf[0,:]

inf_am = updates_inf[2,:]*updates_inf[1,:]
uninf_am = updates_uninf[2,:]*updates_uninf[1,:]

inf_hrm = inf_hrm[np.nonzero(inf_hrm)]
uninf_hrm = uninf_hrm[np.nonzero(uninf_hrm)]

inf_am = inf_am[np.nonzero(inf_am)]
uninf_am = uninf_am[np.nonzero(uninf_am)]

#uninf_avg = (np.mean(uninf_am) + np.mean(uninf_hrm))/2
#inf_avg = (np.mean(inf_am) + np.mean(inf_hrm))/2

#uninf_am /= uninf_avg
#uninf_hrm /= uninf_avg

#inf_am /= inf_avg
#inf_hrm /= inf_avg

min, max = 0.0, 3.6*prob + (1-prob)*1.0

#plt.scatter(inf_hrm/inf_hrm[-1], inf_am/inf_am[-1], label = "inf")
#plt.scatter(uninf_hrm/uninf_hrm[-1], uninf_am/uninf_am[-1], label = "uninf")

plt.scatter(inf_hrm, inf_am, label = "low_dispersion")
plt.scatter(uninf_hrm, uninf_am, label = "high_dispersion")

#plt.plot(inf_hrm, inf_am, label = "low_dispersion")
#plt.plot(uninf_hrm, uninf_am, label = "high_dispersion")

plt.plot([min, max], [min, max], c = 'k')
plt.xlim([min, max])
plt.ylim([min, max])
plt.legend()

plt.show()

#names = ['hrm', 'am', 'lik', 'pri']
#for n, x in zip(names, updates_inf): plt.plot(x, label = n)
#plt.legend()
#plt.savefig("figs/inf_updates")
#plt.show()

#for n, x in zip(names, updates_uninf): plt.plot(x, label = n)
#plt.legend()
#plt.savefig("figs/uninf_updates")
#plt.show()