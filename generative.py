import numpy as np
import torch
#import torch.distributions as dists
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import def_npgaussian_lp
from utils import def_npgaussian_gradlog
from utils import log_softmax
import models
from torch.distributions import Categorical
from torch.distributions import Normal
reload(models)

torch.manual_seed(1)


class Urn ():
    
    @staticmethod
    def E_VI_loss_function(yval, target):
        p = torch.exp(target)/torch.sum(torch.exp(target))
        q = torch.exp(yval)
        
        KL = torch.mm(torch.log(q) - torch.log(p), q.view(-1,1))
            
        return KL

    
    @staticmethod
    def VI_loss_function(yval, target):
        nsamps = 50
        ELBOs = []
        logq = yval.view(1,-1)
        logp = target.view(1,-1)
        #logp = torch.log(torch.exp(target)/torch.sum(torch.exp(target)).view(1,-1))
        
        count = 0
        dist_q = Categorical(torch.exp(logq))
        while count < nsamps:
            s = dist_q.sample().type(torch.LongTensor)            
            #ELBO = logp.index_select(1,s)
            ELBO = logq.index_select(1,s) * (logp.index_select(1,s) - logq.index_select(1,s)/2)
            #print(p.data.numpy() - q.data.numpy(),s.data.numpy(),ELBO.data.numpy())
            ELBOs.append(ELBO)
            count += 1
            
        
        qentropy = 0
        #qentropy = torch.sum(torch.exp(logq) * logq)
        loss = -(torch.mean(torch.cat(ELBOs)) - qentropy)
            
    
        return loss
    
    
    @staticmethod
    def VI_loss_function_grad(yval, target):
        '''
        TODOs:
        1. check dimensionality of q for replications in while loop
-            s = np.random.multinomial(1, q, 1)
-            onehot = np.reshape(s, (-1,1))
-            ELBO_grad += np.reshape(q-s, (1,-1))*(np.dot(logp, onehot) - np.dot(logq, onehot))
+            onehot = np.zeros(2)
+            s = np.reshape(np.random.multinomial(1, np.reshape(q, (-1))), (-1,1))
+            onehot[s[0][0]] = 1
+            ELBO_grad += np.reshape(q[0]-onehot, (1,-1))*(np.dot(logp, onehot) - np.dot(logq, onehot))
        '''
        nsamps = 50
        ELBOs = []
        logq = log_softmax(yval.view(1,-1).data.numpy())
        logp = target.view(1,-1).data.numpy()
        q = np.exp(logq)[0]
        L = logp.shape[1]
        count = 0
        ELBO_grad = 0
        while count < nsamps:
            s = np.random.multinomial(1, q, 1)
            onehot = np.reshape(s, (-1,1))
            ELBO_grad += np.reshape(q-s, (1,-1))*(np.dot(logp, onehot) - np.dot(logq, onehot))
            count += 1
            
        
        grad = ELBO_grad/count
    
        return autograd.Variable(torch.Tensor(grad).type(torch.FloatTensor).view(1,-1)   )

       
    
    def get_approxmodel(self, NUM_LABELS, INPUT_SIZE, nhid, nonlin, stronginit = False):
        
        return models.MLP_disc(INPUT_SIZE, NUM_LABELS, nhid, None, Urn.VI_loss_function_grad, nonlin, stronginit)
    
    
        #return models.MLP_disc(INPUT_SIZE, 2, nhid, loss_function = nn.KLDivLoss())
        

    
    def data_gen(self, block_vals, N_trials, fixed = False, same_urn = False, return_urns = False, variable_ss = None):
        
        '''
        TODO:
        1. Check dimensionality differences in ls from replications vs NU
        Change -- have NU provide a 2-D input as well
        2. is delN sufficient or do we need N too? for PE
        '''
        ps, l1s, l2s = block_vals
        
        N_blocks = len(ps)        
    
        draws = np.empty(shape = (N_trials*N_blocks, 1))
        lik1s = np.empty(shape = (N_trials*N_blocks, 1))
        lik2s = np.empty(shape = (N_trials*N_blocks, 1))
        pris = np.empty(shape = (N_trials*N_blocks, 1))
        Ns = np.empty(shape = (N_trials*N_blocks, 1))
        true_urns = np.empty(shape = (N_trials*N_blocks))
        
        
        for i, (p,l1,l2) in enumerate(zip(ps, l1s, l2s)):
            
            if same_urn:
                urn_b = np.random.binomial(1, p, 1)*np.ones(N_trials)
            else:
                urn_b = np.random.binomial(1, p, N_trials)
             
            draws_b = []
            delNs = [0]
            delN = 0
            for urn in urn_b:
                if urn:
                    lik = l1
                else:
                    lik = l2
                
                if variable_ss is not None:                        
                    draws_b.append(0.0)
                    heads = np.random.binomial(variable_ss[i], lik, 1)
                    tails = variable_ss[i] - heads
                    delN = (heads - tails)[0]
                    delNs.append(delN)            
                        
                else:
                    draw = np.random.binomial(1, lik, 1)
                    draws_b.append(draw)
                    #success = draw*(1-urn) + (1-draw)*urn
                    if same_urn:
                        delN += (2*draw - 1)[0]
                    else:
                        delN = 0
                    delNs.append(delN)
            
            
            if fixed: draws_b = np.ones(N_trials)
            delNs = np.array(delNs, dtype = np.float)
            if variable_ss is not None:
                if variable_ss[i] == 0:
                    Ns[i*N_trials : (i+1)*N_trials, 0] = 0.0
                else: 
                    Ns[i*N_trials : (i+1)*N_trials, 0] = delNs[1]/variable_ss[i]
            else:
                Ns[i*N_trials : (i+1)*N_trials, 0] = 1.0*delNs[:-1] / N_trials              
            draws[i*N_trials : (i+1)*N_trials, 0] = draws_b
            lik1s[i*N_trials : (i+1)*N_trials, 0] = l1 * np.ones(N_trials)
            lik2s[i*N_trials : (i+1)*N_trials, 0] = l2 * np.ones(N_trials)            
            pris[i*N_trials : (i+1)*N_trials, 0] = p * np.ones(N_trials)
            true_urns[i*N_trials : (i+1)*N_trials] = urn_b
            
        
        if variable_ss is not None:
            normalized_ss = 1.0*variable_ss / max(variable_ss)
            X = np.hstack((draws, lik1s, lik2s, pris, Ns, 
                           normalized_ss.reshape((-1, 1))
                           ))
        else:
            X = np.hstack((draws, lik1s, lik2s, pris, Ns, np.ones(shape = (N_trials*N_blocks, 1))))
        X = torch.from_numpy(X)
        X = X.type(torch.FloatTensor)
        
        
        
        if return_urns:
            return X, true_urns
        
        return X
            
    def data_gen_GT(self, N_blocks):
        '''
        Fixed observations
        '''
        #X = np.hstack((draws, lik1s, lik2s, pris, N_ratio, N_t))
        X_options = np.zeros((12, 6))
        X_options[:, :4] = np.array([0.0, 0.4, 0.6, 0.5])
        #Nh, Nt, N
        raw_options = np.array([
            [2, 1, 3],
            [3, 0, 3],
            [3, 2, 5],
            [4, 1, 5],
            [5, 0, 5],
            [5, 4, 9],
            [6, 3, 9],
            [7, 2, 9],
            [9, 8, 17],
            [10, 7, 17],
            [11, 6, 17],
            [19, 14, 33]
        ])
        N_ratio = 1.0*(raw_options[:, 0] - raw_options[:, 1])/raw_options[:, 2]
        N_t = raw_options[:, 2]/33.0
        X_options[:, -2] = N_ratio
        X_options[:, -1] = N_t
        inverse_X = X_options.copy()
        inverse_X[:, 1] = X_options[:, 2].copy()
        inverse_X[:, 2] = X_options[:, 1].copy()
        inverse_X[:, -2] = inverse_X[:, -2]*-1
        
        all_options = np.vstack((X_options, inverse_X))
        
        choices = np.random.choice(np.arange(12), N_blocks)
        X = all_options[choices, :]
        
        X = torch.from_numpy(X)
        X = X.type(torch.FloatTensor)
        
        return X
    
    
    def data_gen_Benj(self, N_blocks, which = None):
        '''
        Fixed observations
        '''
        
        if which == 'GT92':            
            #X = np.hstack((draws, lik1s, lik2s, pris, N_ratio, N_t))
            raw_options = np.array([
                [2, 1, 3],
                [3, 0, 3],
                [3, 2, 5],
                [4, 1, 5],
                [5, 0, 5],
                [5, 4, 9],
                [6, 3, 9],
                [7, 2, 9],
                [9, 8, 17],
                [10, 7, 17],
                [11, 6, 17],
                [19, 14, 33]
            ])
            L = raw_options.shape[0]  
            X_options = np.zeros((L, 6))
            X_options[:, :4] = np.array([-1.0, 0.4, 0.6, 0.5])
            #Nh, Nt, N
            
            N_ratio = 1.0*(raw_options[:, 0] - raw_options[:, 1])/raw_options[:, 2]
            N_t = raw_options[:, 2]/33.0
            X_options[:, -2] = N_ratio
            X_options[:, -1] = N_t
            inverse_X = X_options.copy()
            inverse_X[:, 1] = X_options[:, 2].copy()
            inverse_X[:, 2] = X_options[:, 1].copy()
            inverse_X[:, -2] = inverse_X[:, -2]*-1
            
            all_options = np.vstack((X_options, inverse_X))
            
            choices = np.random.choice(np.arange(L), N_blocks)
            X = all_options[choices, :]
            X = torch.from_numpy(X)
            X = X.type(torch.FloatTensor)
            maxN = 33.0
            
        if which == 'BH80':
            lik0s = 0.2*np.ones(N_blocks)
            lik1s = 1.0 - lik0s
            options = [15.0, 85.0]
            priors = np.tile(options, N_blocks/2)/100.0
            block_vals = (priors, lik0s, lik1s)
            
            X = self.data_gen(block_vals, 1, fixed = True)
            X[:, 4] = (2*X[:, 0] - 1)
            X[:, 0] = -1  
            maxN = 1.0
            
        if which == 'PM65':
            block_vals =  self.assign_PL_replications(6, N_blocks, 'PM', False, False)
            X = self.data_gen(block_vals[:-1], 1)
            X[:, 4] = (2*X[:, 0] - 1)
            X[:, 0] = -1  
            maxN = 1.0
            
        if which == 'DD74':
            
            #X = np.hstack((draws, lik1s, lik2s, pris, N_ratio, N_t))
            #Nh, Nt, N
            raw_options = np.array([
                [0.45, 0.55, 0.5, -1, 1],
                #[0.55, 0.45, 0.5, 1, 1],
                #[0.88, 0.12, 0.5, 1, 0.5],
                [0.12, 0.88, 0.5, -1, 0.5],
                [0.45, 0.55, 0.5, 1, 1],
                #[0.55, 0.45, 0.5, -1, 1],
                #[0.88, 0.12, 0.5, -1, 0.5],
                [0.12, 0.88, 0.5, 1, 0.5]                
                
            ])
            L = raw_options.shape[0]
            X_options = np.zeros((L, 6))
            X_options[:, 0] = -1.0
            
            X_options[:, 1:] = raw_options
            
            choices = np.random.choice(np.arange(L), N_blocks)
            X = X_options[choices, :]
            X = torch.from_numpy(X)
            X = X.type(torch.FloatTensor)  
            maxN = 4
        if which == 'BWB70':
            SS = np.random.choice([4, 5, 6, 7, 8, 9, 10], N_blocks)
            priors = np.random.choice([0.5], N_blocks)
            L1s = np.random.choice([0.7, 0.8], N_blocks)
            L2s = 1.0 - L1s
            block_vals = (priors, L1s, L2s)
            X = self.data_gen(block_vals, 1, variable_ss = SS)
            X[:, 0] = -1  
            maxN = 10
        if which == 'GHR65':
            SS = np.random.choice([1, 3, 6, 9], N_blocks)
            priors = np.random.choice([0.5], N_blocks)
            L1s = np.random.choice([0.6, 0.8], N_blocks)
            L2s = 1.0 - L1s
            block_vals = (priors, L1s, L2s)
            X = self.data_gen(block_vals, 1, variable_ss = SS)
            X[:, 0] = -1  
            maxN = 9 
        if which == 'Gr92':
            SS = np.random.choice([6], N_blocks)
            priors = np.random.choice([0.5, 0.66], N_blocks)
            L1s = np.random.choice([0.5714286], N_blocks)
            L2s = 1.0 - L1s
            block_vals = (priors, L1s, L2s)
            X = self.data_gen(block_vals, 1, variable_ss = SS)
            X[:, 0] = -1  
            maxN = 6              
        if which == 'HS09':
            SS = np.random.choice([0, 1, 2, 3, 4], N_blocks)
            priors = np.random.choice([0.5, 0.66], N_blocks)
            L1s = np.random.choice([0.66], N_blocks)
            L2s = 1.0 - L1s
            block_vals = (priors, L1s, L2s)
            X = self.data_gen(block_vals, 1, variable_ss = SS)
            X[:, 0] = -1  
            maxN = 4
        #if which == 'KW04-1':
            ##X = np.hstack((draws, lik1s, lik2s, pris, N_ratio, N_t))
            #X_options = np.zeros((16, 6))
            #X_options[:, 0] = -1.0
            ##Nh, Nt, N
            #raw_options = np.array([
                #[0.4, 0.6, 0.5, -1, 1.0/25],
                #[0.6, 0.4, 0.5, 1, 1.0/25],
                #[0.6, 0.4, 0.5, 1, 5.0/25],
                #[0.4, 0.6, 0.5, -1, 5.0/25],
                #[0.4, 0.6, 0.5, -1, 15.0/25],
                #[0.6, 0.4, 0.5, 1, 15.0/25],
                #[0.4, 0.6, 0.5, -1, 25.0/25],
                #[0.6, 0.4, 0.5, 1, 25.0/25],
                #[0.4, 0.6, 0.5, 1, 1.0/25],
                #[0.6, 0.4, 0.5, -1, 1.0/25],
                #[0.6, 0.4, 0.5, -1, 5.0/25],
                #[0.4, 0.6, 0.5, 1, 5.0/25],
                #[0.4, 0.6, 0.5, 1, 15.0/25],
                #[0.6, 0.4, 0.5, -1, 15.0/25],
                #[0.4, 0.6, 0.5, 1, 25.0/25],
                #[0.6, 0.4, 0.5, -1, 25.0/25]
                
            #])
            #L = raw_options.shape[0]
            #X_options = np.zeros((L, 6))
            #X_options[:, 0] = -1.0
            
            #X_options[:, 1:] = raw_options
            
            #choices = np.random.choice(np.arange(L), N_blocks)
            #X = X_options[choices, :]
            #X = torch.from_numpy(X)
            #X = X.type(torch.FloatTensor)  
            #maxN = 25
        if which == 'KW04':
            priors = np.random.choice([0.5], N_blocks)
            tempL1 = np.array([80, 60, 52])/100.0
            tempL2 = 60*np.ones(len(tempL1))/100.0
            L1_choices = tempL1 /(tempL1 + tempL2)
            L1s = np.random.choice(L1_choices, N_blocks)
            L2s = 1.0 - L1s
            block_vals = (priors, L1s, L2s)
            X = self.data_gen(block_vals, 1)
            X[:, 4] = (2*X[:, 0] - 1)
            X[:, 0] = -1  
            maxN = 1      
        if which == 'MC72':
            #X = np.hstack((draws, lik1s, lik2s, pris, N_ratio, N_t))
            #Nh, Nt, N
            raw_options = np.array([
                [0.4, 0.6, 0.5, 2/10.0],
                [0.4, 0.6, 0.5, 3/10.0],
                [0.4, 0.6, 0.5, 4/10.0],
                [0.4, 0.6, 0.5, 5/10.0],
                [0.4, 0.6, 0.5, 6/10.0],
                [0.4, 0.6, 0.5, 7/10.0],
                [0.4, 0.6, 0.5, 8/10.0],
                #[0.6, 0.4, 0.5, 2/10.0],
                #[0.6, 0.4, 0.5, 3/10.0],
                #[0.6, 0.4, 0.5, 4/10.0],
                #[0.6, 0.4, 0.5, 5/10.0],
                #[0.6, 0.4, 0.5, 6/10.0],
                #[0.6, 0.4, 0.5, 7/10.0],
                #[0.6, 0.4, 0.5, 8/10.0],
                
            ])
            L = raw_options.shape[0]
            X_options = np.zeros((2*L, 6))
            X_options[:, 0] = -1.0
            X_options[:, -1] = 1.0            
            raw_options_neg = raw_options.copy()
            raw_options_neg[:,-1] = -1 * raw_options[:,-1]
            X_options[:, 1:5] = np.vstack([raw_options, raw_options_neg])
            
            choices = np.random.choice(np.arange(2*L), N_blocks)
            X = X_options[choices, :]
            X = torch.from_numpy(X)
            X = X.type(torch.FloatTensor)  
            maxN = 10
            
        if which == 'Ne01':
            #X = np.hstack((draws, lik1s, lik2s, pris, N_ratio, N_t))
            X_options = np.zeros((20, 6))
            X_options[:, 0] = -1.0
            X_options[:, -1] = 1.0
            #Nh, Nt, N
            raw_options = np.array([
                [0.6, 0.4, 0.5, 5/17.0, 1.0],
                [0.6, 0.4, 0.5, 3/17.0, 1.0],
                [0.6, 0.4, 0.5, 1/17.0, 1.0],
                #[0.4, 0.6, 0.5, -5/17.0, 1.0],
                #[0.4, 0.6, 0.5, -3/17.0, 1.0],
                #[0.4, 0.6, 0.5, -1/17.0, 1.0],
                [0.6, 0.4, 0.5, 3/3.0, 3.0/17.0], 
                [0.6, 0.4, 0.5, 1/3.0, 3.0/17.0],
                #[0.4, 0.6, 0.5, -3/3.0, 3.0/17.0], 
                #[0.4, 0.6, 0.5, -1/3.0, 3.0/17.0]                
                
            ])
            L = raw_options.shape[0]
            X_options = np.zeros((2*L, 6))
            X_options[:, 0] = -1.0
            raw_options_neg = raw_options.copy()
            raw_options_neg[:,-2] = -1 * raw_options[:,-2]
            X_options[:, 1:] = np.vstack([raw_options, raw_options_neg])
            
            
            choices = np.random.choice(np.arange(2*L), N_blocks)
            X = X_options[choices, :]
            X = torch.from_numpy(X)
            X = X.type(torch.FloatTensor)  
            maxN = 17
    
        if which == 'PSM65':
            SS = np.random.choice([1, 4, 12, 48], N_blocks)
            priors = np.random.choice([0.5], N_blocks)
            L1s = np.random.choice([0.6], N_blocks)
            L2s = 1.0 - L1s
            block_vals = (priors, L1s, L2s)
            X = self.data_gen(block_vals, 1, variable_ss = SS)
            X[:, 0] = -1            
            maxN = 48
        
        if which == 'SK07':
            SS = np.random.choice([1, 2, 3, 4], N_blocks)
            priors = np.random.choice([0.5], N_blocks)
            L1s = np.random.choice([0.6], N_blocks)
            L2s = 1.0 - L1s
            block_vals = (priors, L1s, L2s)
            X = self.data_gen(block_vals, 1, variable_ss = SS)
            X[:, 0] = -1  
            maxN = 4     
            
        return X, maxN 


    
    
    def assign_PL_EU(self, N_balls, N_blocks, inf_data):
        
        # but we don't want all of one color ever
        uninf = [4.0, 5.0, 5.0, 6.0]
        inf = [1.0, 2.0, 2.0, 3.0, 7.0, 8.0, 8.0, 9.0]
        
        if inf_data:
            priors = np.random.choice(uninf, N_blocks)/10.0
            lik0s = np.random.choice(inf, N_blocks)/10.0          
            lik1s = 1.0 - lik0s
        else:
            priors = np.random.choice(inf, N_blocks)/10.0
            lik0s = np.random.choice(uninf, N_blocks)/10.0
            lik1s = 1.0 - lik0s
            
        
        return priors, lik0s, lik1s
    
    def assign_PL_FC(self, N_balls, N_blocks, varied):
        

        lik0s = 0.2*np.ones(N_blocks)
        lik1s = 1.0 - lik0s
        options = [15.0, 85.0]
        
        if varied:
            #priors = np.random.choice(options, N_blocks)/N_balls
            priors = np.tile(options, N_blocks/2)/N_balls           
        else:
            p = np.random.choice(options)
            priors = p*np.ones(N_blocks)/N_balls            
        
        return priors, lik0s, lik1s

    def assign_PL_CP(self, N_blocks, N_balls, alpha_post, alpha_pre):
        
            
        posts = np.random.beta(alpha_post, alpha_post, N_blocks)
        pres = np.random.beta(alpha_pre, alpha_pre, N_blocks)#0.5*np.ones(N_blocks)
        priors = []
        likls = []
        
        for pre, post in zip(pres, posts):
            if np.abs(pre - post) > 0.5:
                pre = 1.0 - pre
            x = (post*(1.0 - pre))/(pre*(1.0 - post))
            edit = x / (1.0 + x)
            
            ep = np.clip(np.round(edit*N_balls), 1, N_balls - 1)
            pp = np.clip(np.round(pre*N_balls), 1, N_balls - 1)
            if (np.random.uniform() > 0.0):
                priors.append(pp*1.0 / N_balls)
                likls.append([ep*1.0 / N_balls, 1.0 - ep*1.0 / N_balls])
            else:
                priors.append(ep*1.0 / N_balls)
                likls.append([pp*1.0 / N_balls, 1.0 - pp*1.0 / N_balls])                
                
        
        return np.array(priors).reshape((-1,1)), np.array(likls).reshape((-1,2))[:, 0], np.array(likls).reshape((-1,2))[:, 1] 
    
    def assign_PL_CS(self, N_blocks, N_balls, alpha_post, alpha_prior):
            
        if alpha_post is None:
            priors = np.random.beta(alpha_prior, alpha_prior, N_blocks)
            likls = 0.5*np.ones((N_blocks, 2))
        else:
            likls = []
            priors = []  
            if alpha_post > 1.0:
                posts = np.random.beta(alpha_post, 1.0, N_blocks)
            else:
                posts = np.random.beta(1.0, 1.0/alpha_post, N_blocks)
            ps = np.random.beta(alpha_prior, alpha_prior, N_blocks)
            
            for prior, post in zip(ps, posts):
                if np.abs(prior - post) > 0.5:
                    prior = 1.0 - prior
                x = (post*(1.0 - prior))/(prior*(1.0 - post))
                edit = x / (1.0 + x)
                ep = np.clip(np.round(edit*N_balls), 1, N_balls - 1)
                pp = np.clip(np.round(prior*N_balls), 1, N_balls - 1)
                if (np.random.uniform() > 0.5):
                    priors.append(pp*1.0 / N_balls)
                    likls.append([ep*1.0 / N_balls, 1.0 - ep*1.0 / N_balls])
                else:
                    priors.append(ep*1.0 / N_balls)
                    likls.append([pp*1.0 / N_balls, 1.0 - pp*1.0 / N_balls])                
            
            priors = np.array(priors).reshape((-1,1))
            likls = np.array(likls).reshape((-1,2))
         
        
        return priors, likls[:, 0], likls[:, 1] 

    def assign_PL_replications(self, N_balls, N_blocks, expt_name, fix_prior = False, fix_ll = False):
        
        if expt_name == "PM":
            Ps = np.linspace(0.1,0.9,9)
            LRs = np.array([[3.0,2.0], [4.0,2.0], [5.0,2.0], [5.0,1.0]])/(1.0*N_balls)
            
            LRs = LRs*N_balls
            row_sums = LRs.sum(axis=1)
            LRs = LRs / row_sums[:, np.newaxis]   
            

        elif expt_name == "PE":
            Ps = np.array([0.5])
            LRs = np.array([[85.0, 15.0], [70.0,30.0], [55.0,45.0]])/(1.0*N_balls) 
            
        elif expt_name == "MW":
            Ps = np.array([0.02, 0.05, 0.10, 0.20, 0.98, 0.95, 0.90, 0.80])
            LRs = np.array([[0.4, 0.6], [0.25,0.75], [0.1,0.9]])
            
        if fix_prior:
            Ps = Ps*0 + 0.5
        if fix_ll:
            LRs = LRs*0 + 0.5
            
        
        priors = np.random.choice(Ps, N_blocks)
        l_inds = np.random.choice(np.arange(len(LRs)), N_blocks)
        which_urn = np.random.choice([1,-1], N_blocks)
        
        likls0 = LRs[l_inds]
        likls = np.array([l[::wu] for l,wu in zip(likls0, which_urn)])
    
        return priors.reshape((-1)), likls.reshape((-1,2))[:, 0], likls.reshape((-1,2))[:, 1], l_inds
    

    def assign_PL_GT(self, N_blocks, which):
        
        if which == 'study1':
            Ps = np.array([0.5])
            LRs = np.array([[0.4, 0.6]]) 
                
            
        priors = np.random.choice(Ps, N_blocks)
        l_inds = np.random.choice(np.arange(len(LRs)), N_blocks)
        which_urn = np.random.choice([1,-1], N_blocks)
        
        likls0 = LRs[l_inds]
        likls = np.array([l[::wu] for l,wu in zip(likls0, which_urn)])
        
        return priors.reshape((-1)), likls.reshape((-1,2))[:, 0], likls.reshape((-1,2))[:, 1], l_inds
        

    def assign_PL_demo(self, N_blocks, eq_prior = False, bias = False):
            
            Ps = np.linspace(0.001,0.999,999)
            LRs = np.vstack((Ps, 1.0 - Ps)).T
            if eq_prior:
                Ps = np.array([0.5])
            
            
            priors = np.random.choice(Ps, N_blocks)
            l_inds = np.random.choice(np.arange(len(LRs)), N_blocks)
            which_urn = np.random.choice([1,-1], N_blocks)
            
            
            likls0 = LRs[l_inds]
            likls = np.array([l[::wu] for l,wu in zip(likls0, which_urn)])
            
            if bias:
                likls = likls.reshape((-1,2))
                post = priors*likls[:, 0] / (priors*likls[:, 0] + (1.0 - priors)*likls[:, 1])
                
                switch_prob = np.clip(2*post - 1.0, 0.0, 1.0)
                s = np.random.binomial(np.ones(N_blocks, dtype = 'int'), switch_prob)
                priors = s*priors + (1-s)*(1.0-priors)
                likls[:, 0] = s*likls[:, 0] + (1-s)*(1.0-likls[:, 0])
                likls[:, 1] = s*likls[:, 1] + (1-s)*(1.0-likls[:, 1])                
        
            return priors.reshape((-1)), likls.reshape((-1,2))[:, 0], likls.reshape((-1,2))[:, 1], l_inds
       
    
    def get_rationalmodel(self, N_trials):
        
        return models.UrnRational(N_trials)
    

class Button ():
    
    @staticmethod
    def E_VI_loss_function(yval, target):
        # Exact KL divergence
        # target are the exact posteriors
        
        qmu, qlsd = yval.view(-1,1)
        qlsd = 0.0 * qlsd / qlsd
        pmu, plsd = target.view(-1,1)
    
        
        qsig, psig = (torch.exp(qlsd), torch.exp(plsd))

        #KL = torch.log(psig/qsig) + (qsig**2 + (qmu - pmu)**2)/(2*psig**2) - 1/2
        KL = (qmu - pmu)**2
        
    
        return KL    
    
    @staticmethod
    def VI_loss_function(yval, target):
        nsamps = 100
        # Here the target are the prior and likl params
        # yval are the params given
        # Works only without sigma?
        qmu, qlsd = yval.view(-1)
        qlsd = 0.0 * torch.exp(qlsd) / torch.exp(qlsd)

        prmu, prlsd = target[0,:].view(-1)
        likmu, liklsd = target[1,:].view(-1)  
                
        dist_q = Normal(qmu, torch.exp(qlsd))
        dist_pr = Normal(prmu, torch.exp(prlsd))
        dist_lik = Normal(likmu, torch.exp(liklsd))
        
        count = 0
        ELBOs = []
        while count < nsamps:
            s = dist_q.sample()
            #ELBO = dist_lik.log_prob(s) + dist_pr.log_prob(s)
            ELBO = dist_q.log_prob(s) * (dist_lik.log_prob(s) + dist_pr.log_prob(s) - dist_q.log_prob(s)/2)
            ELBOs.append(ELBO)
            count += 1
            
            
        qentropy = 0
        #qentropy = 0.5 + 0.5 * torch.log(2 * np.pi) + torch.log(self.scale)
        qentropy = qlsd
            
        loss = -torch.log((torch.mean(torch.cat(ELBOs)) + qentropy))
        
        #print(torch.mean(torch.cat(ELBOs)).data.numpy(), qentropy.data.numpy(), "***")
        
        if np.isnan(loss.data.numpy()): 
            raise ValueError ("Loss is NaN")
        
    
        return loss
    
    @staticmethod
    def VI_loss_function_grad(yval, target):
        nsamps = 80
        yval = yval.view(-1).data.numpy()
        target = target.view(2,-1).data.numpy()
        
        qmu, qlsd = yval
                
        dist_q = def_npgaussian_lp(yval)
        dist_pr = def_npgaussian_lp(target[0,:])
        dist_lik = def_npgaussian_lp(target[1,:])
        gradq = def_npgaussian_gradlog(yval)
        
        #print('Q ', yval[0], np.exp(yval[1]))
        #print('prior ', target[0, 0], np.exp(target[0, 1]))
        #print('likelihood ', target[1, 0], np.exp(target[1, 1]))
        
        count = 0
        while count < nsamps:
            s = np.random.normal(qmu, np.exp(qlsd))
            if np.isinf(target[1,1]):
                val = gradq(s) * (dist_pr(s) - dist_q(s))
            else:
                val = gradq(s) * (dist_lik(s) + dist_pr(s) - dist_q(s))
            if count == 0 :
                ELBO_grad = val
            else:
                ELBO_grad += val
            count += 1
        
        grad = ELBO_grad/count
        grad = np.clip(grad, -10, 10)
        #print(grad)
        #print('***********')
        
    
        return autograd.Variable(torch.Tensor(grad).type(torch.FloatTensor).view(1,-1)   )
    
    
    def get_approxmodel(self, DIM, INPUT_SIZE, nhid, nonlin):
        return models.MLP_cont(INPUT_SIZE, DIM, nhid, None, Button.VI_loss_function_grad, nonlin)
    
    def data_gen(self, ps, lik_var, N_trials):
        
        lik_sd = np.sqrt(lik_var)
        N_blocks = len(ps)
        
        last = np.empty(shape = (N_trials*N_blocks, 1))
        m_so_far = np.empty(shape = (N_trials*N_blocks, 1))
        Ns = np.empty(shape = (N_trials*N_blocks, 1))
        
        tvals = np.empty(shape = (N_trials*N_blocks, 1))
        for i, p in enumerate(ps):
            
            vals = np.random.normal(p, lik_sd, N_trials)
            #prvs = vals.copy()
            #prvs[1:] = prvs[:-1]
            #prvs[0] = 0
            
            N_b = (1.0*np.arange(N_trials)) + 1.0 
            #N_b = (1.0*(np.arange(N_trials) + 1.0))/N_trials
            
            msf = np.cumsum(vals)/(N_b)
            #msf[1:] = msf[:-1]
            #msf[0] = 0
                
            last[i*N_trials : (i+1)*N_trials, 0] = vals#prvs
            m_so_far[i*N_trials : (i+1)*N_trials, 0] = msf
            Ns[i*N_trials : (i+1)*N_trials, 0] = N_b
            
            tvals[i*N_trials : (i+1)*N_trials, 0] = vals            
            
        
        
        X = np.hstack((last, m_so_far, Ns))
        X = torch.from_numpy(X)
        X = X.type(torch.FloatTensor)
        
        #Y = torch.from_numpy(tvals).view(-1,1)
        #Y = Y.type(torch.FloatTensor)
        
        
        return X
    
    
    
    def assign_PL(self, N_blocks, fac):
        pr_mu = 0
        
        pr_sd = np.sqrt(fac)
        
        priors = np.random.normal(pr_mu, pr_sd, N_blocks)

        return priors
    
        
    
    def get_rationalmodel(self, prior_var, lik_var, N_trials):
        
        return models.ButtonRational(prior_var, lik_var, N_trials)
    