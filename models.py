import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class MLPClassifier(nn.Module): 

    def __init__(self, num_labels, input_size, nhid):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, nhid)
        self.fc2 = nn.Linear(nhid, num_labels)
        self.loss_function = nn.NLLLoss()

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.log_softmax(x)
        return x
    
    def train(self, data, N_epoch, optimizer):
        
        for epoch in range(N_epoch):
            for x, y in zip(data["train"]["X"], data["train"]["y"]):
        
                self.zero_grad()
        
                target = autograd.Variable(y)
                log_probs = self(autograd.Variable(x)).view(1,-1)
        
                loss = self.loss_function(log_probs, target)
                loss.backward()
                optimizer.step()
                
    def test (self, data):
        # validate approx_models - will come back to this for resource rationality
        err_prob = 0    
        err = 0 
        count = 0.0
        print("\n*********************\n")
        pred = []
        for x, y in zip(data["X"], data["y"]):
            log_probs = self(autograd.Variable(x)).view(1,-1)
            err_prob += np.exp(log_probs.data.numpy()[0][1 - y.numpy()[0]])
            err += round(np.exp(log_probs.data.numpy()[0][1 - y.numpy()[0]]))
            pred.append(np.exp(log_probs.data.numpy()[0][0]))
            count += 1.0
        
        data["y_pred"] = np.array(pred)        
        err /= count
        err_prob /= count
        print("classification error : {0}, \
        with prob : {1}".format(round(100*err), round(100*err_prob)))
        print("\n*********************\n")
        
    

class hier_model():
    
    def __init__(self):
        self.holder = 0
        

class block_model():
    
    def __init__(self):
        self.holder = 0