import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.special import beta

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
                
    def test (self, data, sg_epoch, name):
        # validate approx_models - will come back to this for resource rationality
        err_prob = 0    
        err = 0 
        count = 0.0
        pred = []
        for x, y in zip(data["X"], data["y"]):
            log_probs = self(autograd.Variable(x)).view(1,-1)
            err_prob += np.exp(log_probs.data.numpy()[0][1 - y.numpy()[0]])
            err += round(np.exp(log_probs.data.numpy()[0][1 - y.numpy()[0]]))
            pred.append(np.exp(log_probs.data.numpy()[0][1]))
            count += 1.0
            
            datapoint = {"X": x.view(1, -1),
                         "y": y.view(1, -1)}
            self.train(datapoint, sg_epoch)
        
        data["y_pred" + name] = np.array(pred)        
        err /= count
        err_prob /= count
        print("classification error : {0}, \
        with prob : {1}".format(round(100*err), round(100*err_prob)))
        print("*********************")
        
    

class MLPRegressor(nn.Module): 

    def __init__(self, input_size, nhid):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, nhid)
        self.fc2 = nn.Linear(nhid, 1)
        self.loss_function = nn.MSELoss()
        return

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x
    
    def train(self, data, N_epoch):
        
        for epoch in range(N_epoch):
            for x, y in zip(data["X"], data["y"]):
        
                self.zero_grad()
        
                target = autograd.Variable(y)
                yval = self(autograd.Variable(x)).view(1,-1)
        
                loss = self.loss_function(yval, target)
                loss.backward()
                self.optimizer.step()
        return
                
    def test (self, data, sg_epoch):
        # validate approx_models - will come back to this for resource rationality
        err = 0 
        count = 0.0
        pred = []
        for x, y in zip(data["X"], data["y"]):
            yval = self(autograd.Variable(x)).view(1,-1)
            err += (yval.data.numpy() - y)**2
            pred.append(yval.data.numpy()[0])
            count += 1.0
            
            datapoint = {"X": x.view(1, -1),
                         "y": y.view(1, -1)}
            self.train(datapoint, sg_epoch)
        
        data["y_pred"] = np.array(pred)        
        err /= count
        err = sqrt(err)
        print("MS error : {0}".format(round(100*err)))
        print("*********************")
    
        return
        
    

class UrnRational():
        
    def __init__(self, prior_fac, N_trials):
        self.alpha = prior_fac
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
        

        # Brute force MLE on point estimates of mu
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

        for x, y in zip(data["X"], data["y"]):
            count += 1                
            draw, lik, pri, N = x.numpy()
            urn = y.numpy()
            #print("X", x.numpy())
            #pred = self.pred_post(draw, lik, pri, N)
            if not count%self.N_t:
                self.update_params(pri, N, urn)
            
        return    
    
    def test (self, data, name):
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
            #print(pred)
            count += 1.0
            
            #datapoint = {"X": x.view(1, -1),
                         #"y": y.view(1, -1)}
            #self.train(datapoint)
        
        data["y_pred" + name] = np.array(preds).flatten()       
        err /= count
        err_prob /= count
        print("classification error : {0}, \
        with prob : {1}".format(round(100*err), round(100*err_prob)))
        print("*********************")
        return
        
        
    
class ButtonRational():
        
    def __init__(self):
        self.holder = 0
        