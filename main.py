import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import generative
import utils

import matplotlib.pyplot as plt

torch.manual_seed(60)

N_epoch = 30
sg_epoch = 0
    
N_blocks = 20
N_trials = 20
N_balls = 9
testN_blocks = 600
valN_blocks = 2

fac = 10.0
prior_fac = 1
NUM_LABELS = 2
DIM = 1

INPUT_SIZE = 4
nhid = 1
    
expts = {"disc": {},
         "cont": {}}

expts["disc"]["expt_type"] = generative.Urn()
expts["cont"]["expt_type"] = generative.Button()

L2 = {"disc": {'inf_p': 0, 'uninf_p':0},
      "cont": {'inf_p': 0.0, 'uninf_p':0.0}}

for expt in ['cont']:
    
    expts[expt]["data"] = {}
    expts[expt]["data"]["inf_p"] = {}
    expts[expt]["data"]["uninf_p"] = {}
    expts[expt]["am"] = {}
    expts[expt]["hrm"] = {}
    
    for cond in expts[expt]["data"]:
        expts[expt]["data"][cond]["train"] = {}
        expts[expt]["data"][cond]["val"] = {}
        expts[expt]["data"][cond]["test"] = {}
        if expt == 'disc':
            expts[expt]["am"][cond] = expts[expt]["expt_type"].get_approxmodel(NUM_LABELS, INPUT_SIZE, nhid)
        elif expt == 'cont':
            expts[expt]["am"][cond] = expts[expt]["expt_type"].get_approxmodel(DIM, INPUT_SIZE, nhid)
            
        expts[expt]["hrm"][cond] = expts[expt]["expt_type"].get_rationalmodel(prior_fac, N_trials)
        
        # Generate data
        
        if (cond == "uninf_p"):
            
            ps, ls = expts[expt]["expt_type"].assign_PL(N_balls, N_blocks, fac)
            expts[expt]['data'][cond]['train']['ps'], expts[expt]['data'][cond]['train']['ls'] = np.repeat(ps, N_trials), np.repeat(ls, N_trials)

            expts[expt]["data"][cond]["train"]["X"],  expts[expt]["data"][cond]["train"]["y"] = \
                expts[expt]["expt_type"].data_gen(ps, ls, N_trials, N_blocks, N_balls)
            
            ps, ls = expts[expt]["expt_type"].assign_PL(N_balls, valN_blocks, fac)
            expts[expt]['data'][cond]['val']['ps'], expts[expt]['data'][cond]['val']['ls'] = \
                np.repeat(ps, N_trials), np.repeat(ls, N_trials)
            
            expts[expt]["data"][cond]["val"]["X"],  expts[expt]["data"][cond]["val"]["y"] = \
                expts[expt]["expt_type"].data_gen(ps, ls, N_trials, valN_blocks, N_balls)        
            
            ps, ls = expts[expt]["expt_type"].assign_PL(N_balls, testN_blocks, 1.0/fac)
            expts[expt]['data'][cond]['test']['ps'], expts[expt]['data'][cond]['test']['ls'] = \
                np.repeat(ps, N_trials), np.repeat(ls, N_trials)
            
            expts[expt]["data"][cond]["test"]["X"],  expts[expt]["data"][cond]["test"]["y"] = \
                expts[expt]["expt_type"].data_gen(ps, ls, N_trials, testN_blocks, N_balls)
        else:
            
            ps, ls = expts[expt]["expt_type"].assign_PL(N_balls, N_blocks, 1.0/fac)
            expts[expt]['data'][cond]['train']['ps'], expts[expt]['data'][cond]['train']['ls'] = \
                np.repeat(ps, N_trials), np.repeat(ls, N_trials)
            
            expts[expt]["data"][cond]["train"]["X"],  expts[expt]["data"][cond]["train"]["y"] = \
                expts[expt]["expt_type"].data_gen(ps, ls, N_trials, N_blocks, N_balls)
            
            ps, ls = expts[expt]["expt_type"].assign_PL(N_balls, valN_blocks, 1.0/fac)
            expts[expt]['data'][cond]['val']['ps'], expts[expt]['data'][cond]['val']['ls'] = \
                np.repeat(ps, N_trials), np.repeat(ls, N_trials)
            
            expts[expt]["data"][cond]["val"]["X"],  expts[expt]["data"][cond]["val"]["y"] = \
                expts[expt]["expt_type"].data_gen(ps, ls, N_trials, valN_blocks, N_balls)
    
            ps, ls = expts[expt]["expt_type"].assign_PL(N_balls, testN_blocks, fac)
            expts[expt]['data'][cond]['test']['ps'], expts[expt]['data'][cond]['test']['ls'] = \
                np.repeat(ps, N_trials), np.repeat(ls, N_trials)
            
            expts[expt]["data"][cond]["test"]["X"],  expts[expt]["data"][cond]["test"]["y"] = \
                expts[expt]["expt_type"].data_gen(ps, ls, N_trials, testN_blocks, N_balls)
            
        
            
        # train models

        # Block model doesn't get trained
        # Hierarchical model trains it's prior over mu_p and mu_l
        d = expts[expt]["data"][cond]["train"]
        d = expts[expt]["hrm"][cond].train(d)

        # approx
        expts[expt]["am"][cond].optimizer = \
                        optim.SGD(expts[expt]["am"][cond].parameters(), lr=0.01, weight_decay = L2[expt][cond])
        
        try:
            utils.load_model(expts[expt]["am"],
                             cond, N_epoch, sg_epoch, 
                             round(fac), N_blocks, N_trials, expt, prefix = 'aftertest_')
        except IOError:
            expts[expt]["am"][cond].train(d, N_epoch)
            
            utils.save_model(expts[expt]["am"], cond, N_epoch, sg_epoch, round(fac), 
                             N_blocks, N_trials, expt, prefix = '')
                
        
    #for cond in expts[expt]["data"]:
    
        
        
        expts[expt]["am"][cond].optimizer = \
            optim.SGD(expts[expt]["am"][cond].parameters(), lr=0.01)
        
                
        print(cond, "Val")
        dset = "val"
        d = expts[expt]["data"][cond][dset]
        print("Hierarchical rational model gives: ")
        d = expts[expt]["hrm"][cond].test(d)
        
        print("Approximate model gives: ")

        d = expts[expt]["am"][cond].test(d, sg_epoch, N_trials)
        
        print(cond, "Test")
        dset = "test"
        
        d = expts[expt]["data"][cond][dset]
        print("Hierarchical rational model gives: ")
        d = expts[expt]["hrm"][cond].test(d)
        
        
        #try:
            #utils.load_model(expts[expt]["am"],
                             #cond, N_epoch, sg_epoch, 
                             #round(fac), N_blocks, N_trials, expt, prefix = 'aftertest_')
        #except IOError:
        print("Approximate model gives: ")
        d = expts[expt]["am"][cond].test(d, sg_epoch, N_trials)
        utils.save_model(expts[expt]["am"], cond, N_epoch, sg_epoch, round(fac), 
                         N_blocks, N_trials, expt, prefix = 'aftertest_')
    
        
        
        
    
                    
    #if testN_blocks < 7:
    #utils.plot_both(expts[expt]["data"], 'hrm', "val", expt, fac, N_epoch)
    #utils.plot_both(expts[expt]["data"], 'hrm', "test", expt, fac, N_epoch)
    
    #utils.plot_both(expts[expt]["data"], 'am', "val", expt, fac, N_epoch)
    #utils.plot_both(expts[expt]["data"], 'am', "test", expt, fac, N_epoch)
    
        

    #else:
    di = expts[expt]["data"]["inf_p"]['test']
    du = expts[expt]["data"]["uninf_p"]['test']
    utils.plot_calibration(di, du, N_epoch, sg_epoch, round(fac), N_blocks, N_trials, expt)
