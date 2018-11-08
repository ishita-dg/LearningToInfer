import numpy as np
import torch
#import torch.distributions as dists
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

    def __init__(self, input_size, output_size, nhid, loss_function, loss_function_grad):
        super(MLP_disc, self).__init__()
        self.out_dim = output_size
        self.fc1 = nn.Linear(input_size, nhid)
        self.fc2 = nn.Linear(nhid, output_size)
        self.loss_function = loss_function
        self.loss_function_grad = loss_function_grad
        return

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        #x = F.log_softmax(x)
        return x
    
    def def_newgrad(self, yval, target):
            y = self.loss_function_grad(yval, target)
            self.newgrad = lambda x : y    
    
    def train(self, data, N_epoch, verbose = True):
        
        for epoch in range(N_epoch):
            if not epoch%10 and verbose: print("Epoch number: ", epoch)            
            for x, y in zip(data["X"], data["log_joint"]):
        
                self.zero_grad()
        
                target = autograd.Variable(y)
                yval = self(autograd.Variable(x)).view(1,-1)
            
                
                if self.loss_function is not None:
                    loss = self.loss_function(yval, target)
                    self.newgrad = lambda x : x
                    
                elif self.loss_function_grad is not None:
                    loss = nn.MSELoss()(yval, target.view(1,-1))
                    #loss = nn.MSELoss()(yval, target)
                    self.def_newgrad(yval, target)
                
                yval.register_hook(self.newgrad)
                loss.backward()
                self.optimizer.step()
        
        return
    
    def test (self, data, sg_epoch, N_trials, nft = True, name = None):

        count = 0.0
        pred = []
        datapoint = {}
        
        orig = copy.deepcopy(self)
        
        for x, lj in zip(data["X"], data["log_joint"]):
            if not (count)%(N_trials * 25) : print("Testing, ", count/N_trials)
            
            if (not datapoint.keys() or nft):
                datapoint = {"X": x.view(1, -1),
                             "log_joint": lj.view(1,-1)}                
            else:
                datapoint["X"] = torch.cat((datapoint["X"], x.view(1, -1)), 0)
                datapoint["log_joint"] = torch.cat((datapoint["y_log_joint"], lj.view(1, -1)), 0)

            if (not count%N_trials):
                self = copy.deepcopy(orig)
                datapoint = {}
            else:
                self.train(datapoint, sg_epoch, verbose = False)
                
            yval = self(autograd.Variable(x)).view(1,-1)
            yval = F.log_softmax(yval, dim = 1)
            pred.append(np.exp(yval.data.numpy())[0])
            count += 1.0
            
                
            
        pred0 = torch.from_numpy(np.array(pred)).view(-1, self.out_dim)
        data["y_am"] = pred0.type(torch.FloatTensor)
        
        return data
        
        


class MLP_cont(nn.Module): 

    def __init__(self, input_size, output_size, nhid, loss_function, loss_function_grad):
        super(MLP_cont, self).__init__()
        self.fc1 = nn.Linear(input_size, nhid)
        #self.fc_opt = nn.Linear(nhid, nhid)
        self.fc2 = nn.Linear(nhid, output_size)
        self.loss_function = loss_function
        self.loss_function_grad = loss_function_grad
        return

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
       
        x = self.fc2(x)
        return x
    
    def def_newgrad(self, yval, target):
        y = self.loss_function_grad(yval, target)
        self.newgrad = lambda x : y
        
    def train(self, data, N_epoch, verbose = True):
        
        for epoch in range(N_epoch):
            if not epoch%10 and verbose: print("Epoch number: ", epoch)
            #for x, y in zip(data["X"], data["y_hrm"]):
            for x, y in zip(data["X"], data["log_joint"]):  
                
                self.zero_grad()
                
                yval = self(autograd.Variable(x)).view(1,-1)
        
                if self.loss_function is not None:
                    loss = self.loss_function(yval, y)
                    self.newgrad = lambda x : x
                    
                elif self.loss_function_grad is not None:
                    loss = nn.MSELoss()(yval, y.view(1,-1)[0,:2])
                    self.def_newgrad(yval, y)
                
                yval.register_hook(self.newgrad)
                loss.backward()
                self.optimizer.step()
        return
                
    def test (self, data, sg_epoch, N_trials, nft = True, name = None):

        count = 0.0
        pred = []
        datapoint = {}
        
        orig = copy.deepcopy(self)
        
        for x, y_p, lj in zip(data["X"], data["y_hrm"], data["log_joint"]):
            
            if not (count)%(N_trials * 25) : print("Testing, ", count/N_trials)
            count += 1.0
            
            if (not datapoint.keys() or nft):
                datapoint = {"X": x.view(1, -1),
                             "y_hrm": y_p.view(1, -1),
                             "log_joint": lj.view(1,2,2)}                
            else:
                datapoint["X"] = torch.cat((datapoint["X"], x.view(1, -1)), 0)
                datapoint["y_hrm"] = torch.cat((datapoint["y_hrm"], y_p.view(1, -1)), 0)
                datapoint["log_joint"] = torch.cat((datapoint["y_log_joint"], lj.view(1,2,2)), 0)

            if (not count%N_trials):
                self = copy.deepcopy(orig)
                datapoint = {}
            else:
                self.train(datapoint, sg_epoch, verbose = False)
                
            yval = self(autograd.Variable(x)).view(1,-1)
            pred.append(yval.data.numpy()[0])
            
                
            
        pred0 = torch.from_numpy(np.array(pred)).view(-1,2)
        data["y_pred_am"] = pred0.type(torch.FloatTensor)
        
        return
        
    

class UrnRational():
        
    def __init__(self, N_trials):
        self.N_t = N_trials
        self.mus = []
        return
    
    def pred_loglik(self, draw, lik, N):
        '''
        TODO: Ensure that NU has the right lik convention
        '''
        N += (2*draw - 1)
        if (N == 0):
            return(np.array([0.0, 0.0]))
        sign = int(N/np.abs(N))
        likl = (lik[::sign])**abs(N)
        likl /= sum(likl)
        return np.log(likl)
    
    def pred_logprior(self, pri, N):
        # point estimate of posterior mu_p
        p0 = pri
        return np.log(np.clip(np.array([1.0 - p0, p0]), 0.01, 0.99))
    
    def log_joint(self, draw, lik, pri, N):
        return self.pred_loglik(draw, lik, N) + self.pred_logprior(pri, N)
        
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
        

        ## Brute force MLE on point estimates of alpha
        #pri *= N
        #n_in = (N + pri)*self.N_t/2.0 + urn
        #self.mus.append(n_in*1.0)
        
        
        
        #x = np.arange(0.0, 50.0, 0.01)
        #y = self.alpha_ll(x)
        #self.alpha = x[np.argmax(y)]
        
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

        for x in data["X"]:
            count += 1                
            draw, lik1, lik2, pri, N = x.numpy()
            lik = np.array([lik1, lik2])
            preds.append(self.pred_post(draw, lik, pri, N))
            ljs.append(self.log_joint(draw, lik, pri, N))
        
        pred0 = torch.from_numpy(np.array(preds)).view(-1,2)
        data["y_hrm"] = pred0.type(torch.FloatTensor)
        
        lj0 = torch.from_numpy(np.array(ljs)).view(-1,2)
        data["log_joint"] = lj0.type(torch.FloatTensor)
        
        return data
    
    def test (self, data, name = None):
        # validate approx_models - will come back to this for resource rationality
        err = 0 
        err_prob = 0
        count = 0.0
        preds = []
        ljs = []
        
        for x in data["X"]:
            draw, lik1, lik2, pri, N = x.numpy()
            lik = np.array([lik1, lik2])
            pred = self.pred_post(draw, lik, pri, N)
            preds.append(pred)
            ljs.append(self.log_joint(draw, lik, pri, N))
            
        pred0 = torch.from_numpy(np.array(preds)).view(-1,2)
        data["y_hrm"] = pred0.type(torch.FloatTensor)
        
        lj0 = torch.from_numpy(np.array(ljs)).view(-1,2)
        data["log_joint"] = lj0.type(torch.FloatTensor)

        return data
        
        
    
class ButtonRational():
        
    def __init__(self, prior_var, lik_var, N_trials):
        self.pr_mu = 0
        self.lik_var = lik_var
        self.pr_var = prior_var
        self.N_t = N_trials
        self.data_mus = []
    
    def update_params(self, msf, block_n, block_vals):
        # MLE for variance across blocks
        #self.mus.append(msf)
        #if len(self.mus) > 1 : self.prior_var = np.var(np.array(self.mus))
        #self.lik_var += (np.var(np.array(block_vals)) - self.lik_var)/block_n
        return
    
    def log_joint(self, last, msf, N):
        msf = autograd.Variable(torch.Tensor(np.array([msf]))).view(1,-1)
        pr_mu = autograd.Variable(torch.Tensor([self.pr_mu])).view(1,-1)
        llik_var = autograd.Variable(
            torch.Tensor(np.array([np.log(self.lik_var/(N*self.N_t))]))).view(1,-1)
        lpr_var = autograd.Variable(
            torch.Tensor(np.array([np.log(self.pr_var)]))).view(1,-1)
        
        #print(self.lik_var, N*self.N_t, llik_var.data.numpy())
        
        pr_vec = torch.cat((pr_mu, lpr_var/2), 0).view(1,-1)
        lik_vec = torch.cat((msf, llik_var/2), 0).view(1,-1)
        lj_vec = torch.cat((pr_vec, lik_vec),0)
             
        return lj_vec        
    
    #def log_joint(self, last, msf, N):
        #msf = autograd.Variable(torch.Tensor(np.array([msf]))).view(1,-1)
        #m = autograd.Variable(torch.Tensor([self.pr_mu])).view(1,-1)
        #s2 = autograd.Variable(torch.Tensor([self.lik_var])).view(1,-1)
        #v2 = autograd.Variable(torch.Tensor(np.array([self.prior_var]))).view(1,-1)
                
        #pri_log_prob = lambda x : -(((x - msf)/torch.sqrt(s2))**2 
                                    #+ torch.log(2*np.pi*s2))/2.0
        
        #likl_log_prob = lambda x : -(((x - m)/torch.sqrt(v2))**2 
                                            #+ torch.log(2*np.pi*v2))/2.0
        
        ##likl = dists.Normal(msf, torch.sqrt(s2))
        ##pri = dists.Normal(self.pr_mu, torch.sqrt(v2))
        
        #lj_func = lambda x: pri_log_prob(x) + likl_log_prob(x)         
        #return lj_func
        
    
    def pred_post_MAP(self, last, msf, N):
        est_mu = (self.lik_var*self.pr_mu + self.pr_var*(msf * N * self.N_t)) / \
            (self.pr_var * N * self.N_t + self.lik_var)
        
        est_sig = 1.0/\
            np.sqrt(N * self.N_t/self.lik_var + 1.0/self.pr_var)
        return np.array([est_mu, np.log(est_sig)])
    
    def train(self, data):
        
        count = 0
        preds = []
        f_joints = []
        block_vals = []

        for x in data["X"]:
            count += 1                
            last, msf, N = x.numpy()
            preds.append(self.pred_post_MAP(last, msf, N))
            f_joints.append(self.log_joint(last, msf, N))
            block_vals.append(last)
            if not count%self.N_t:
                self.update_params(msf, count/self.N_t, block_vals)
                block_vals = []
        
        pred0 = torch.from_numpy(np.array(preds)).view(-1,2)
        data["y_hrm"] = autograd.Variable(pred0.type(torch.FloatTensor))
        
        data["log_joint"] = f_joints
        
        return data
    
    def test (self, data, name = None):
        # validate approx_models - will come back to this for resource rationality
        preds = []
        f_joints = []
        
        for x in data["X"]:
            last, msf, N = x.numpy()
            pred = self.pred_post_MAP(last, msf, N)
            preds.append(pred)
            f_joints.append(self.log_joint(last, msf, N))
            
        pred0 = torch.from_numpy(np.array(preds)).view(-1,2)
        data["y_hrm"] = autograd.Variable(pred0.type(torch.FloatTensor))
        
        data["log_joint"] = f_joints
        
         
        return data
            
    
    