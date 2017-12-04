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
    
    def train(self, data, N_epoch):
        
        for epoch in range(N_epoch):
            for x, y in zip(data["X"], data["y"]):
        
                self.zero_grad()
        
                target = autograd.Variable(y)
                log_probs = self(autograd.Variable(x)).view(1,-1)
        
                loss = self.loss_function(log_probs, target)
                loss.backward()
                self.optimizer.step()
                
    def test (self, data, sg_epoch, nft = True, name = None):
        # validate approx_models - will come back to this for resource rationality
        err_prob = 0    
        err = 0 
        count = 0.0
        pred = []
        datapoint = {}
        for x, y in zip(data["X"], data["y"]):
            log_probs = self(autograd.Variable(x)).view(1,-1)
            err_prob += np.exp(log_probs.data.numpy()[0][1 - y.numpy()[0]])
            err += round(np.exp(log_probs.data.numpy()[0][1 - y.numpy()[0]]))
            pred.append(np.exp(log_probs.data.numpy()[0][1]))
            count += 1.0
            
            if (not datapoint.keys() or nft):
                datapoint = {"X": x.view(1, -1),
                             "y": y.view(1, -1)}                
            else:
                datapoint["X"] = torch.cat((datapoint["X"], x.view(1, -1)), 0)
                datapoint["y"] = torch.cat((datapoint["y"], y.view(1, -1)), 0)

            
            self.train(datapoint, sg_epoch)
        
        pred0 = torch.from_numpy(logit(np.array(pred)).flatten())
        data["y_pred_am"] = pred0.type(torch.FloatTensor)
        err /= count
        err_prob /= count
        print("classification error : {0}, \
        MSE error : {1}".format(round(100*err), -1))
        print("*********************")
        
        
    

class MLPRegressor(nn.Module): 

    def __init__(self, input_size, output_size, nhid):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, nhid)
        self.fc2 = nn.Linear(nhid, output_size)
        self.loss_function = nn.MSELoss()
        return

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x
    
    def train(self, data, N_epoch):
        
        for epoch in range(N_epoch):
            for x, y in zip(data["X"], data["y_pred_hrm"]):
        
                self.zero_grad()
        
                target = autograd.Variable(y)
                yval = self(autograd.Variable(x)).view(1,-1)
        
                loss = self.loss_function(yval, target)
                loss.backward()
                self.optimizer.step()
        return
                
    def test (self, data, sg_epoch, N_trials, nft = True, name = None):
        # validate approx_models - will come back to this for resource rationality
        err_cl = 0 
        err_mse = 0
        count = 0.0
        pred = []
        datapoint = {}
        
        orig = copy.deepcopy(self)
        
        for x, y, y_p in zip(data["X"], data["y"], data["y_pred_hrm"]):
            yval = self(autograd.Variable(x)).view(1,-1)
            pred.append(yval.data.numpy()[0][0])
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
                
            py = inv_logit(yval.data.numpy())[0][0]
            ty = 1 - y.numpy()[0]
            err_cl += round(ty*py + (1-ty)*(1 -py)) 
            err_mse += (y.numpy() - yval.data.numpy()[0][0])**2
            
        
        pred0 = torch.from_numpy(np.array(pred).flatten())
        data["y_pred_am"] = pred0.type(torch.FloatTensor)
        err_cl /= count
        err_mse /= count
        print("classification error : {0}, \
        MSE error : {1}".format(round(100*err_cl), err_mse))
        print("*********************")
        
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
        pri *= N
        n_in = (N + pri)*self.N_t/2.0
        n_out = (N - pri)*self.N_t/2.0
        alpha0 = n_in + self.alpha
        beta0 = n_out + self.alpha
        p0 = alpha0/(alpha0 + beta0)
        #self.pr
        return np.log(np.clip(np.array([1.0 - p0, p0]), 0.01, 0.99))
    
    def pred_post(self, draw, lik, pri, N):
        # draw is 0 if col is one that is more un urn 0
        # lik if the prob of col 0 in urn 0
        p = self.pred_loglik(draw, lik) + self.pred_logprior(pri, N)
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

        for x, y in zip(data["X"], data["y"]):
            count += 1                
            draw, lik, pri, N = x.numpy()
            urn = y.numpy()
            preds.append(self.pred_post(draw, lik, pri, N)[1])
            if not count%self.N_t:
                self.update_params(pri, N, urn)
        
        pred0 = torch.from_numpy(logit(np.array(preds))).view(-1,1)
        data["y_pred_hrm"] = pred0.type(torch.FloatTensor)
        
        
        return data
    
    def test (self, data, name = None):
        # validate approx_models - will come back to this for resource rationality
        err = 0 
        err_prob = 0
        count = 0.0
        preds = []
        
        for x, y in zip(data["X"], data["y"]):
            draw, lik, pri, N = x.numpy()
            urn = y.numpy()
            pred = self.pred_post(draw, lik, pri, N)[1]
            preds.append(pred)
            err_prob += self.pred_post(draw, lik, pri, N)[1 - urn]
            err += round(self.pred_post(draw, lik, pri, N)[1 - urn])
            count += 1.0
            
        pred0 = torch.from_numpy(logit(np.array(preds))).view(-1,1)
        data["y_pred_hrm"] = pred0.type(torch.FloatTensor)
        
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
    
    def pred_MAP(self, last, msf, N):
        est_mu = (self.s2*self.m + self.v2*(msf * N * self.N_t)) / \
            (self.v2 * N * self.N_t + self.s2)
        
        est_sig = 1.0/\
            np.sqrt(N * self.N_t/self.s2 + 1.0/self.v2)
        return np.array([est_mu, est_sig])
    
    def train(self, data):
        
        count = 0
        preds = []

        for x, y in zip(data["X"], data["y"]):
            count += 1                
            last, msf, N, _ = x.numpy()
            preds.append(self.pred_MAP(last, msf, N))
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
            pred = self.pred_MAP(last, msf, N)
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
            
    
    