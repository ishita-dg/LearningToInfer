import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import models
reload(models)

torch.manual_seed(1)


class Urn ():
    
    def get_approxmodel(self, NUM_LABELS, INPUT_SIZE, nhid):
        
        return models.MLPRegressor(INPUT_SIZE, nhid)
        #return models.MLPClassifier(NUM_LABELS, INPUT_SIZE, nhid)

    
    def data_gen(self, ps, ls, N_trials, N_blocks, N_balls):
    
        draws = np.empty(shape = (N_trials*N_blocks, 1))
        liks = np.empty(shape = (N_trials*N_blocks, 1))
        pris = np.empty(shape = (N_trials*N_blocks, 1))
        Ns = np.empty(shape = (N_trials*N_blocks, 1))
        
        urns = np.empty(shape = (N_trials*N_blocks, 1))
        
        # we want to remove overall bias for urn 1 vs 0
        
        # let the color that is in excess in urn 1
        # be assigned index 1
        
        # lik represents the fraction of col1 in urn1
        
        # pri is the fraction of times the left urn is chosen in one block
        # minus the number of times the right urn is chosen
        
        
        for i, (p,l) in enumerate(zip(ps, ls)):
            l = l/N_balls

            urn_b = np.random.binomial(1, p, N_trials)
            
            # diff between N1 and N0
            # high when many in urn 1
            pri_b = np.cumsum(2*urn_b - 1)
            pri_b[1:] = pri_b[:-1]
            pri_b[0] = 0
            N_b = (1.0*np.arange(N_trials))/N_trials
            #N_b = np.zeros(N_trials)
            pri_b = pri_b / (np.arange(N_trials) + 1.0)
            
            # probability of 
            if l < 0.5 : l = 1.0 - l
            draws_b = []
            for urn in urn_b:
                if urn:
                    lik = l
                else:
                    lik = 1.0-l
                # lik will always be the higher prob
                # we want to draw col 0 with this prob
                draw = np.random.binomial(1, lik, 1)
                draws_b.append(draw)
                
            draws[i*N_trials : (i+1)*N_trials, 0] = draws_b
            urns[i*N_trials : (i+1)*N_trials, 0] = urn_b
            liks[i*N_trials : (i+1)*N_trials, 0] = l * np.ones(N_trials)
            pris[i*N_trials : (i+1)*N_trials, 0] = pri_b
            Ns[i*N_trials : (i+1)*N_trials, 0] = N_b
            
        
        X = np.hstack((draws, liks, pris, Ns))
        X = torch.from_numpy(X)
        X = X.type(torch.FloatTensor)
        
        Y = torch.from_numpy(urns)
        Y = Y.type(torch.LongTensor)
        
        
        return(X, Y)
    
    
    
    def assign_PL(self, N_balls, N_blocks, fac):
        
        # but we don't want all of one color ever
        
    
        alpha_p = beta_p = fac
        priors = np.round(100*np.random.beta(alpha_p, beta_p, N_blocks))/100
        priors = np.clip(priors, 0.1, 0.9)
        
        alpha_l = beta_l = 1.0/fac
        likls = np.round(N_balls*np.random.beta(alpha_l, beta_l, N_blocks))
        likls = np.clip(likls, 1, N_balls - 1)
        
        return priors, likls
            
        
    
    def get_rationalmodel(self, prior_fac, N_trials):
        
        return models.UrnRational(prior_fac, N_trials)
    

class Button ():
    
    def get_approxmodel(self, INPUT_SIZE, nhid):
        return models.MLPRegressor(INPUT_SIZE, nhid)
    
    def data_gen(self, ps, ls, N_trials, N_blocks, N_balls):
    
        draws = np.empty(shape = (N_trials*N_blocks, 1))
        liks = np.empty(shape = (N_trials*N_blocks, 1))
        pris = np.empty(shape = (N_trials*N_blocks, 1))
        
        urns = np.empty(shape = (N_trials*N_blocks, 1))
        
        # we want to remove overall bias for urn 1 vs 0
        # let the urn on the left be assigned index 0
        
        # let the color that is in excess in the urn on the left
        # be assigned index 0
        
        # lik represents the fraction of col0 in urn0
        
        # pri is the fraction of times the left urn is chosen in one block
        # minus the number of times the right urn is chosen
        
        
        for i, (p,l) in enumerate(zip(ps, ls)):
            l = l/N_balls
            # the probability of "high prior" 
            # being on left or right is the same
            pr = np.random.choice([p, 1.0 - p])
            urn_b = np.random.binomial(1, pr, N_trials)
            
            pri_b = np.cumsum(- 2*urn_b + 1)
            pri_b[1:] = pri_b[:-1]
            pri_b[0] = 0
            pri_b = pri_b / (np.arange(N_trials) + 1.0)
            
            # probability of 
            if l < 0.5 : l = 1.0 - l
            draws_b = []
            for urn in urn_b:
                if not urn:
                    lik = l
                else:
                    lik = 1.0-l
                # lik will always be the higher prob
                # we want to draw col 0 with this prob
                draw = 1.0 - np.random.binomial(1, lik, 1)
                draws_b.append(draw)
                
            draws[i*N_trials : (i+1)*N_trials, 0] = draws_b
            urns[i*N_trials : (i+1)*N_trials, 0] = urn_b
            liks[i*N_trials : (i+1)*N_trials, 0] = l * np.ones(N_trials)
            pris[i*N_trials : (i+1)*N_trials, 0] = pri_b
            
        
        X = np.hstack((draws, liks, pris))
        X = torch.from_numpy(X)
        X = X.type(torch.FloatTensor)
        
        Y = torch.from_numpy(urns)
        Y = Y.type(torch.LongTensor)
        
        
        return(X, Y)
    
    
    
    def assign_PL(self, N_balls, N_blocks, var_fac, mean_fac_prior, mean_fac_likl):
    
        alpha_p = var_fac
        beta_p = var_fac*mean_fac_prior
        priors = np.round(100*np.random.beta(alpha_p, beta_p, N_blocks))/100
        
        alpha_l = var_fac
        beta_l = var_fac*mean_fac_likl
        likls = np.round(N_balls*np.random.beta(alpha_l, beta_l, N_blocks))
        
        return priors, likls
            
        
    
    def get_rationalmodel(self):
        
        return models.ButtonRational(prior_var_fac)
    