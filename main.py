import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import generative

torch.manual_seed(1)

class MLPClassifier(nn.Module): 

    def __init__(self, num_labels, input_size, nhid):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, nhid)
        self.fc2 = nn.Linear(nhid, num_labels)

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.log_softmax(x)
        return x
loss_function = nn.NLLLoss()
N_epoch = 150    
    
# Urn example
N_blocks = 15
N_trials = 20
N_balls = 9
testN_blocks = int(N_blocks/5)
valN_blocks = int(N_blocks/5)
    

data = {}
data["inf_p"] = {}
data["uninf_p"] = {}
models = {}

var_fac = 1000
mean_fac = 0.25
NUM_LABELS = 2
INPUT_SIZE = 3
nhid = 5

for cond in data:
    data[cond]["train"] = {}
    data[cond]["val"] = {}
    data[cond]["test"] = {}
    models[cond] = MLPClassifier(NUM_LABELS, INPUT_SIZE, nhid)
    
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
    
    optimizer = optim.SGD(models[cond].parameters(), lr=0.1)
    
    for epoch in range(N_epoch):
        for x, y in zip(data[cond]["train"]["X"], data[cond]["train"]["y"]):
           
            models[cond].zero_grad()
    
            target = autograd.Variable(y)
            log_probs = models[cond](autograd.Variable(x)).view(1,-1)
    
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()
            
    # validate models - will come back to this for resource rationality

    err_val = 0    
    err_test = 0  
    
    print("\n*********************\n")
    pred = []
    for x, y in zip(data[cond]["val"]["X"], data[cond]["val"]["y"]):
        log_probs = models[cond](autograd.Variable(x)).view(1,-1)
        err_val += np.exp(log_probs.data.numpy()[0][1 - y.numpy()[0]])
        pred.append(np.exp(log_probs.data.numpy()[0][0]))
    
    data[cond]["val"]["y_pred"] = np.array(pred)        
    err_val /= N_trials*valN_blocks
    print("Validation loss on condition ", cond, " is ", err_val)
    
    
    # test models
    pred = []
    for x, y in zip(data[cond]["test"]["X"], data[cond]["test"]["y"]):
        log_probs = models[cond](autograd.Variable(x)).view(1,-1)
        err_test += np.exp(log_probs.data.numpy()[0][1 - y.numpy()[0]])
        pred.append(np.exp(log_probs.data.numpy()[0][0]))
        
    data[cond]["test"]["y_pred"] = np.array(pred)
    err_test /= N_trials*testN_blocks
    print("Test loss on condition ", cond, " is ", err_test)
    
        
    print("\n*********************\n")



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
        
        plt.savefig('{0}{1}epochs{2}.png'.format(dset,round(100.0/(1.0 + mean_fac)), N_epoch))
        
        
plot_both("val")
plot_both("test")
    