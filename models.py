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

class RbfNet(nn.Module):
    def __init__(self, centers, num_class=10):
        super(RbfNet, self).__init__()
        self.num_class = num_class
        self.num_centers = centers.size(0)

        self.centers = nn.Parameter(centers)
        self.beta = nn.Parameter(torch.ones(1,self.num_centers)/10)
        self.linear = nn.Linear(self.num_centers, self.num_class, bias=True)
        utils.initialize_weights(self)


    def forward(self, batches):
        radial_val = self.kernel_fun(batches)
        class_score = self.linear(radial_val)
        return class_score



class MLP_disc(nn.Module): 

    def __init__(self, input_size, output_size, nhid, loss_function, loss_function_grad, nonlin, stronginit = False):
        
        super(MLP_disc, self).__init__()
        self.out_dim = output_size
        self.fc1 = nn.Linear(input_size, nhid)
        self.fc2 = nn.Linear(nhid, output_size)
        
        if stronginit:
            fac = 1.5
            self.fc1.weight.data = self.fc1.weight.data*fac
            self.fc2.weight.data = self.fc2.weight.data*fac
            self.fc1.bias.data = self.fc1.bias.data*fac
            self.fc2.bias.data = self.fc2.bias.data*fac
            
        self.loss_function = loss_function
        self.loss_function_grad = loss_function_grad
            
        
        self.num_centers = nhid
        self.centers = nn.Parameter(torch.zeros(self.num_centers))
        
        self.beta = nn.Parameter(torch.ones(self.num_centers)/10)
        self.nonlin = nonlin
        
        return
    
    def kernel_fun(self, x):
        return torch.exp(-self.beta.mul((x-self.centers).pow(2) ) )

    def forward(self, x):
        x = self.fc1(x)
        if 'tanh' in self.nonlin: 
            x = F.tanh(x)
        elif 'rbf' in self.nonlin:
            x = self.kernel_fun(x)

        x = self.fc2(x)
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
                    #loss = nn.MSELoss()(yval, target.view(1,-1)[0,:2])
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

    def __init__(self, input_size, output_size, nhid, loss_function, loss_function_grad, nonlin):
        super(MLP_cont, self).__init__()
        self.out_dim = output_size
        self.fc1 = nn.Linear(input_size, nhid)
        self.fc_opt = nn.Linear(nhid, nhid)
        self.fc2 = nn.Linear(nhid, output_size)
        self.loss_function = loss_function
        self.loss_function_grad = loss_function_grad
        
        self.num_centers = nhid
        self.centers = nn.Parameter(torch.zeros(self.num_centers))
        self.beta = nn.Parameter(torch.ones(self.num_centers)/10)
        self.nonlin = nonlin
        
        return
    
    def kernel_fun(self, x):
        return torch.exp(-self.beta.mul((x-self.centers).pow(2) ) )

    def forward(self, x):
        x = self.fc1(x)
        #x = F.tanh(x)
        #x = self.fc_opt(x)
        if 'tanh' in self.nonlin: 
            x = F.tanh(x)
        elif 'rbf' in self.nonlin:
            x = self.kernel_fun(x)
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
                target = autograd.Variable(y)
        
                if self.loss_function is not None:
                    loss = self.loss_function(yval, y)
                    self.newgrad = lambda x : x
                    
                elif self.loss_function_grad is not None:
                    loss = nn.MSELoss()(yval, target.view(1,-1)[0,:2])
                    self.def_newgrad(yval, target)
                
                yval.register_hook(self.newgrad)
                loss.backward()
                self.optimizer.step()
        return data
                
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
                             "log_joint": lj}                
            else:
                datapoint["X"] = torch.cat((datapoint["X"], x.view(1, -1)), 0)
                datapoint["y_hrm"] = torch.cat((datapoint["y_hrm"], y_p.view(1, -1)), 0)
                datapoint["log_joint"] = torch.cat((datapoint["log_joint"], lj), 0)

            if (not count%N_trials):
                self = copy.deepcopy(orig)
                datapoint = {}
            else:
                self.train(datapoint, sg_epoch, verbose = False)
                
            yval = self(autograd.Variable(x)).view(1,-1)
            pred.append(yval.data.numpy()[0])
            
                
            
        pred0 = torch.from_numpy(np.array(pred)).view(-1,2)
        data["y_am"] = pred0.type(torch.FloatTensor)
        
        return data
        
    

class UrnRational():
        
    def __init__(self, N_trials):
        self.N_t = N_trials
        self.mus = []
        return
    
    def pred_loglik(self, draw, lik, N):
        '''
        TODO: Ensure that NU has the right lik convention
        '''
        if draw is not None:
            N += (2*draw - 1)
        if (N == 0):
            return(np.array([0.0, 0.0]))
        sign = (int(N/np.abs(N)) + 1.0)/2.0
        likl = sign*lik.astype('float64') + (1-sign)*(1.0 - lik.astype('float64'))
        likl = (likl)**abs(N)
        return np.log(likl)
    
    def pred_logprior(self, pri, N):
        p0 = pri
        return np.log(np.clip(np.array([1.0 - p0, p0]), 0.0001, 0.9999))
    
    def log_joint(self, draw, lik, pri, N):
        return self.pred_loglik(draw, lik, N) + self.pred_logprior(pri, N)
        
    def pred_post(self, draw, lik, pri, N):
        # draw is 0 if col is one that is more un urn 0
        # lik if the prob of col 0 in urn 0
        p = self.log_joint(draw, lik, pri, N)
        p = p.astype('float64')
        return np.exp(p)/sum(np.exp(p))
    

    def train(self, data):
        
        count = 0
        preds = []
        ljs = []

        for x in data["X"]:
            count += 1                
            draw, lik1, lik2, pri, N_ratio = x.numpy()
            N = N_ratio*self.N_t
            lik = np.array([lik1, lik2])
            preds.append(self.pred_post(draw, lik, pri, N))
            ljs.append(self.log_joint(draw, lik, pri, N))
        
        pred0 = torch.from_numpy(np.array(preds)).view(-1,2)
        data["y_hrm"] = pred0.type(torch.DoubleTensor)
        
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
            draw, lik1, lik2, pri, N_ratio = x.numpy()
            N = N_ratio*self.N_t
            lik = np.array([lik1, lik2])
            pred = self.pred_post(draw, lik, pri, N)
            preds.append(pred)
            ljs.append(self.log_joint(draw, lik, pri, N))
            
        pred0 = torch.from_numpy(np.array(preds)).view(-1,2)
        data["y_hrm"] = pred0.type(torch.DoubleTensor)
        
        lj0 = torch.from_numpy(np.array(ljs)).view(-1,2)
        data["log_joint"] = lj0.type(torch.FloatTensor)

        return data
        
    def train_GT(self, data, max_N = 33):
            
        count = 0
        preds = []
        ljs = []

        for x in data["X"]:
            count += 1                
            draw, lik1, lik2, pri, N_ratio, N_t = x.numpy()
            N = N_ratio*N_t*max_N
            draw = None
            lik = np.array([lik1, lik2])
            preds.append(self.pred_post(draw, lik, pri, N))
            ljs.append(self.log_joint(draw, lik, pri, N))
        
        pred0 = torch.from_numpy(np.array(preds)).view(-1,2)
        data["y_hrm"] = pred0.type(torch.DoubleTensor)
        
        lj0 = torch.from_numpy(np.array(ljs)).view(-1,2)
        data["log_joint"] = lj0.type(torch.FloatTensor)
        
        return data
    
    def test_GT (self, data, max_N = 33, name = None):
        # validate approx_models - will come back to this for resource rationality
        err = 0 
        err_prob = 0
        count = 0.0
        preds = []
        ljs = []
        
        for x in data["X"]:
            draw, lik1, lik2, pri, N_ratio, N_t = x.numpy()
            N = N_ratio*N_t*max_N
            draw = None
            lik = np.array([lik1, lik2])
            pred = self.pred_post(draw, lik, pri, N)
            preds.append(pred)
            ljs.append(self.log_joint(draw, lik, pri, N))
            
        pred0 = torch.from_numpy(np.array(preds)).view(-1,2)
        data["y_hrm"] = pred0.type(torch.DoubleTensor)
        
        lj0 = torch.from_numpy(np.array(ljs)).view(-1,2)
        data["log_joint"] = lj0.type(torch.FloatTensor)

        return data
            
        
    
class ButtonRational():
        
    def __init__(self, prior_var, lik_var, N_trials):
        self.pr_mu = 0
        self.lik_var = lik_var
        self.pr_var = prior_var
        self.N_t = 1.0 #N_trials -- currently N is not normalized
        self.data_mus = []

    
    def log_joint(self, last, msf, N):
        lj_vec = np.array([[self.pr_mu, np.log(self.pr_var)/2.0], 
                           [msf, np.log(self.lik_var/N)/2.0]])
        return lj_vec 
        
    
    def pred_post_MAP(self, last, msf, N):
        est_mu = (self.lik_var*self.pr_mu + self.pr_var*(msf * N)) / \
            (self.pr_var * N + self.lik_var)
        
        est_sig = 1.0/\
            np.sqrt(N/self.lik_var + 1.0/self.pr_var)
        return np.array([est_mu, np.log(est_sig)])
    
    def train(self, data):
        
        count = 0
        preds = []
        ljs = []
        block_vals = []

        for x in data["X"]:
            count += 1                
            last, msf, N = x.numpy()
            preds.append(self.pred_post_MAP(last, msf, N))
            ljs.append(self.log_joint(last, msf, N))
            block_vals.append(last)
        
        pred0 = torch.from_numpy(np.array(preds)).view(-1,2)
        data["y_hrm"] = pred0.type(torch.DoubleTensor)
        
        lj0 = torch.from_numpy(np.array(ljs)).view(-1,2,2)
        data["log_joint"] = lj0.type(torch.FloatTensor)
        
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
        data["y_hrm"] = pred0.type(torch.DoubleTensor)
        
        data["log_joint"] = f_joints
        
         
        return data
            
    
    