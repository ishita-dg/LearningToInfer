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

N_epoch = 30
sg_epoch = 0
    
N_blocks = 20
N_trials = 10
N_balls = 9
testN_blocks = 200
valN_blocks = 2

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
    

    fac = 100.0
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

        # Block model doesn't get trained
        # Hierarchical model trains it's prior over mu_p and mu_l
        d = expts[expt]["data"][cond]["train"]
        d = expts[expt]["hrm"][cond].train(d)

        # approx
        expts[expt]["am"][cond].optimizer = \
            optim.SGD(expts[expt]["am"][cond].parameters(), lr=0.1)
        expts[expt]["am"][cond].train(d, N_epoch)
                
        
    for cond in expts[expt]["data"]:
                
        print("Val")
        dset = "val"
        d = expts[expt]["data"][cond][dset]
        d = expts[expt]["hrm"][cond].test(d)

        d = expts[expt]["am"][cond].test(d, sg_epoch, N_trials)
        
        print("Test")
        dset = "test"
        
        d = expts[expt]["data"][cond][dset]
        d = expts[expt]["hrm"][cond].test(d)

        d = expts[expt]["am"][cond].test(d, sg_epoch, N_trials)
        
    
                
    if testN_blocks < 4:
        utils.plot_both(expts[expt]["data"], 'hrm', "val", fac, N_epoch)
        utils.plot_both(expts[expt]["data"], 'hrm', "test", fac, N_epoch)
        utils.plot_both(expts[expt]["data"], 'am', "val", fac, N_epoch)
        utils.plot_both(expts[expt]["data"], 'am', "test", fac, N_epoch)
    
        
        

di = expts[expt]["data"]["inf_p"]['test']
du = expts[expt]["data"]["uninf_p"]['test']
utils.plot_calibration(di, du, N_epoch, sg_epoch, round(fac), N_blocks, N_trials)
