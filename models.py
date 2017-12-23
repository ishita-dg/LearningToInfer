import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.special import beta
from utils import inv_logit
from utils import logit
import copy

torch.manual_seed(1)


class MLP_disc(nn.Module): 

    def __init__(self, input_size, output_size, nhid, loss_function):
        super(MLP_disc, self).__init__()
        self.fc1 = nn.Linear(input_size, nhid)
        self.fc2 = nn.Linear(nhid, output_size)
        self.loss_function = loss_function
        return

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.log_softmax(x)
        return x
    
    def train(self, data, N_epoch):
        
        for epoch in range(N_epoch):
            for x, y in zip(data["X"], data["y_joint"]):
        
                self.zero_grad()
        
                target = autograd.Variable(y)
                yval = self(autograd.Variable(x)).view(1,-1)
        
                loss = self.loss_function(yval, target)
                loss.backward()
                self.optimizer.step()
        return
                
    def test (self, data, sg_epoch, N_trials, nft = True, name = None):
        count = 0.0
        pred = []
        datapoint = {}
        
        orig = copy.deepcopy(self)
        
        for x, y, y_p in zip(data["X"], data["y"], data["y_pred_hrm"]):
            yval = self(autograd.Variable(x)).view(1,-1)
            pred.append(yval.data.numpy()[0])
            count += 1.0
            
            if (not datapoint.keys() or nft):
                datapoint = {"X": x.view(1, -1),
                             "y": y.view(1, -1),
                             "y_pred_hrm": y_p.view(1, -1)}                
            else:
                datapoint["X"] = torch.cat((datapoint["X"], x.view(1, -1)), 0)
                datapoint["y"] = torch.cat((datapoint["y"], y.view(1, -1)), 0)
                datapoint["y_pred_hrm"] = torch.cat((datapoint["y_pred_hrm"], y_p.view(1, -1)), 0)

            if (not count%N_trials):
                self = copy.deepcopy(orig)
                datapoint = {}
            else:
                self.train(datapoint, sg_epoch)
            
        
        pred0 = torch.from_numpy(np.exp(np.array(pred))).view(-1,2)
        data["y_pred_am"] = pred0.type(torch.FloatTensor)

        
        return
        


class MLP_cont(nn.Module): 

    def __init__(self, input_size, output_size, nhid, loss_function):
        super(MLP_disc, self).__init__()
        self.fc1 = nn.Linear(input_size, nhid)
        self.fc2 = nn.Linear(nhid, output_size)
        self.loss_function = loss_function
        return

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x
    
    def train(self, data, N_epoch):
        
        for epoch in range(N_epoch):
            for x, y in zip(data["X"], data["y_joint"]):
        
                self.zero_grad()
        
                target = autograd.Variable(y)
                yval = self(autograd.Variable(x)).view(1,-1)
        
                loss = self.loss_function(yval, target)
                loss.backward()
                self.optimizer.step()
        return
                
    def test (self, data, sg_epoch, N_trials, nft = True, name = None):

        count = 0.0
        pred = []
        datapoint = {}
        
        orig = copy.deepcopy(self)
        
        for x, y, y_p in zip(data["X"], data["y"], data["y_pred_hrm"]):
            yval = self(autograd.Variable(x)).view(1,-1)
            pred.append(yval.data.numpy()[0])
            count += 1.0
            
            if (not datapoint.keys() or nft):
                datapoint = {"X": x.view(1, -1),
                             "y": y.view(1, -1),
                             "y_pred_hrm": y_p.view(1, -1)}                
            else:
                datapoint["X"] = torch.cat((datapoint["X"], x.view(1, -1)), 0)
                datapoint["y"] = torch.cat((datapoint["y"], y.view(1, -1)), 0)
                datapoint["y_pred_hrm"] = torch.cat((datapoint["y_pred_hrm"], y_p.view(1, -1)), 0)

            if (not count%N_trials):
                self = copy.deepcopy(orig)
                datapoint = {}
            else:
                self.train(datapoint, sg_epoch)
                
            
        pred0 = torch.from_numpy(np.exp(np.array(pred))).view(-1,2)
        data["y_pred_am"] = pred0.type(torch.FloatTensor)
        
        return
        
    

class UrnRational():
        
    def __init__(self, prior_fac, N_trials):
        self.alpha = prior_fac
        self.N_t = N_trials
        self.mus = []
        return
    
    def pred_loglik(self, draw, lik):
        urn = np.array([0, 1])
        logp = np.zeros(2)
        logp += (draw*urn + (1 - draw)*(1 - urn))*np.log(lik) 
        logp += (draw*(1 - urn) + (1 - draw)*urn)*np.log(1 - lik) 
        #print("logp", logp)
        return logp
    
    def pred_logprior(self, pri, N):
        # point estimate of posterior mu_p
        p0 = pri
        return np.log(np.clip(np.array([1.0 - p0, p0]), 0.01, 0.99))
    
    def log_joint(self, draw, lik, pri, N):
        return self.pred_loglik(draw, lik) + self.pred_logprior(pri, N)
        
    def pred_post(self, draw, lik, pri, N):
        # draw is 0 if col is one that is more un urn 0
        # lik if the prob of col 0 in urn 0
        p = self.log_joint(draw, lik, pri, N)
        return np.exp(p)/sum(np.exp(p))
    
    
    def alpha_ll(self, x):
        mus = np.array(self.mus)
        y = 0
        y += (x - 1)*np.sum((np.log(mus/self.N_t)))
        y += (x - 1)*np.sum((np.log(1.0 - mus/self.N_t)))
        y -= len(self.mus)*np.log(beta(x, x))
        return y
    
    def update_params(self, pri, N, urn):
        

        # Brute force MLE on point estimates of alpha
        pri *= N
        n_in = (N + pri)*self.N_t/2.0 + urn
        self.mus.append(n_in*1.0)
        
        
        
        x = np.arange(0.0, 50.0, 0.01)
        y = self.alpha_ll(x)
        self.alpha = x[np.argmax(y)]
        
        ## method of moments estimators from beta params
       
        #mus = np.array(self.mus)
        #m1 = np.mean(mus)
        #m2 = np.mean(mus**2)
        #n = self.N_t
        
        #self.alpha = (n*m1 - m2)/(m1 + n*(m2/m1 - m1 -1))
        #self.alpha = ((n - m1)*(n - m2/m1))/(m1 + n*(m2/m1 - m1 -1))
        
        return
        

    def train(self, data):
        
        count = 0
        preds = []
        ljs = []

        for x, y in zip(data["X"], data["y"]):
            count += 1                
            draw, lik, pri, N = x.numpy()
            urn = y.numpy()
            preds.append(self.pred_post(draw, lik, pri, N))
            ljs.append(self.log_joint(draw, lik, pri, N))
            if not count%self.N_t:
                self.update_params(pri, N, urn)
        
        pred0 = torch.from_numpy(np.array(preds)).view(-1,2)
        data["y_pred_hrm"] = pred0.type(torch.FloatTensor)
        
        lj0 = torch.from_numpy(np.exp(np.array(ljs))).view(-1,2)
        data["y_joint"] = lj0.type(torch.FloatTensor)
        
        return data
    
    def test (self, data, name = None):
        # validate approx_models - will come back to this for resource rationality
        err = 0 
        err_prob = 0
        count = 0.0
        preds = []
        ljs = []
        
        for x, y in zip(data["X"], data["y"]):
            draw, lik, pri, N = x.numpy()
            urn = y.numpy()
            pred = self.pred_post(draw, lik, pri, N)
            preds.append(pred)
            ljs.append(self.log_joint(draw, lik, pri, N))
            err_prob += self.pred_post(draw, lik, pri, N)[1 - urn]
            err += round(self.pred_post(draw, lik, pri, N)[1 - urn])
            count += 1.0
            
        pred0 = torch.from_numpy(np.array(preds)).view(-1,2)
        data["y_pred_hrm"] = pred0.type(torch.FloatTensor)
        
        lj0 = torch.from_numpy(np.exp(np.array(ljs))).view(-1,2)
        data["y_joint"] = lj0.type(torch.FloatTensor)
        
        err /= count
        err_prob /= count
        print("classification error : {0}, \
        MSE error : {1}".format(round(100*err), -1))
        print("*********************")

        return data
        
        
    
class ButtonRational():
        
    def __init__(self, prior_fac, N_trials):
        self.m = 0
        self.s2 = 1**2
        self.v2 = 10**2
        self.N_t = N_trials
        self.mus = []
    
    def update_params(self, msf):
        # MLE for variance across blocks
        self.mus.append(msf)
        if len(self.mus) > 1 : self.v2 = np.var(np.array(self.mus))
        return
    
    def log_joint(last, msf, N):
        #*TODO*
        return lj
        
    
    def pred_post(self, last, msf, N):
        est_mu = (self.s2*self.m + self.v2*(msf * N * self.N_t)) / \
            (self.v2 * N * self.N_t + self.s2)
        
        est_sig = 1.0/\
            np.sqrt(N * self.N_t/self.s2 + 1.0/self.v2)
        return np.array([est_mu, est_sig])
    
    def train(self, data):
        
        count = 0
        preds = []
        f_joint = []

        for x, y in zip(data["X"], data["y"]):
            count += 1                
            last, msf, N, _ = x.numpy()
            preds.append(self.pred_post(last, msf, N))
            if not count%self.N_t:
                self.update_params(msf)
        
        pred0 = torch.from_numpy(np.array(preds)).view(-1,2)
        data["y_pred_hrm"] = pred0.type(torch.FloatTensor)
        
        
        return data
    
    def test (self, data, name = None):
        # validate approx_models - will come back to this for resource rationality
        err_mse = 0 
        count = 0.0
        preds = []
        
        for x, y in zip(data["X"], data["y"]):
            last, msf, N, _ = x.numpy()
            tmu = y.numpy()
            pred = self.pred_post(last, msf, N)
            preds.append(pred)
            err_mse += (pred - tmu)**2
            count += 1.0
            
        pred0 = torch.from_numpy(np.array(preds)).view(-1,2)
        data["y_pred_hrm"] = pred0.type(torch.FloatTensor)
        
        err_mse /= count
        print("classification error : {0}, \
        MSE error : {1}".format( -1, err_mse))
        print("*********************")
         
        return data
            
    
    