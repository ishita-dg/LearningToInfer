import numpy as np
import torch
#import torch.distributions as dists
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import gaussian_logpdf
import models
reload(models)

torch.manual_seed(1)


class Urn ():
    
    
    @staticmethod
    def VI_loss_function(yval, samples, target):
        
        q = torch.exp(yval)
        
        vals0 = torch.mm(samples, (torch.log(q) * torch.log(target)).view(-1,1))
        vals1 = torch.mm(samples, (torch.log(q) * torch.log(q)).view(-1,1))
    
        ELBO = torch.mean(vals0 - vals1/2)
        
        return -ELBO
    
    
    def get_approxmodel(self, NUM_LABELS, INPUT_SIZE, nhid):
        
        return models.MLP_disc(INPUT_SIZE, 2, nhid, loss_function = Urn.VI_loss_function)
    
    
        #return models.MLP_disc(INPUT_SIZE, 2, nhid, loss_function = nn.KLDivLoss())
        

    
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
            #urns[i*N_trials : (i+1)*N_trials, 0] = urn_b
            urns[i*N_trials : (i+1)*N_trials, 0] = np.zeros(N_trials)
            liks[i*N_trials : (i+1)*N_trials, 0] = l * np.ones(N_trials)
            pris[i*N_trials : (i+1)*N_trials, 0] = p * np.ones(N_trials)
            #pris[i*N_trials : (i+1)*N_trials, 0] = pri_b
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
    
    @staticmethod
    def E_VI_loss_function(yval, target):
        
        qmu, qsig = yval[0,0], torch.exp(yval[0,1])
        pmu, psig = target[0], torch.exp(target[1])
        
        weight = autograd.Variable(torch.Tensor([1,0]), requires_grad = False)
        
        
        KL = torch.log(psig/qsig) +\
            (qsig**2 + (qmu - pmu)**2)/(2*psig**2) -\
            1/2
        
        revKL = torch.log(qsig/psig) +\
            (psig**2 + (pmu - qmu)**2)/(2*qsig**2) -\
            1/2
        
        # Numerical trouble with KL and revKL, switching to log()
        
        JS = torch.log((qsig**2 + (qmu - pmu)**2)/(2*psig**2) +\
            (psig**2 + (pmu - qmu)**2)/(2*qsig**2))
        
        #sqerr_m = torch.sqrt(torch.mean((qmu - pmu)**2))
        sqerr = torch.sqrt(torch.mean(weight*(yval.view(-1) - target)**2))
        
        #print("Errors", revKL, KL)
        
        #print(yval.data, target.data, "_____________")
        
    

        #if np.isnan(sqerr.data.numpy()): 
            #print("************", factors, yval.view(-1), target)
        #if np.isnan(revKL.data.numpy()) or np.isnan(KL.data.numpy()): 
        
        #if np.isnan(JS.data.numpy()): 
            ##print("************", qmu, pmu, qsig, psig, "************")
            #print("************", psig.data, qsig.data, yval.data, JS) 
            #raise ValueError ("nananana")
   
        return JS
    
    #@staticmethod
    #def VI_loss_function(yval, samples, target):
        #'''
        #Takes joint probability function as P input
        #'''
        
        ## Here the target is a lambda function for log joint
        ## yval are the params given
        
        #logq = gaussian_logpdf(yval, samples)
        #logp = target(samples)
        
        #ELBO = torch.mean(logq*(logp - logq/2))
        
        ##print("yval, ", yval.data.numpy())
        #print("samples", samples.data.numpy()[:3])
        #print("logp", logp.data.numpy()[0][:3])
        #print("logq", logq.data.numpy()[0][:3])
        
        #print("loss, ", ELBO.data.numpy())
        #print("**************")
        #return -ELBO
    
    @staticmethod
    def VI_loss_function(yval, samples, target):
        '''
        Takes prior and likl params as P input
        '''
        
        # Here the target is a lambda function for log joint
        # yval are the params given
        
        logq = gaussian_logpdf(yval, samples)
        prior = target[0,:].view(1,-1)
        likl = target[1,:].view(1,-1)
        logp = gaussian_logpdf(prior, samples) + gaussian_logpdf(likl, samples)
        
        #ELBO = torch.mean(logq*(logp - logq/2))

        ELBO = torch.mean(logp - logq)
        #print("yval, ", yval.data.numpy())
        #print("prior and likle", prior.data.numpy(), likl.data.numpy())
        #print("samples", samples.data.numpy()[:3])
        #print("logp", logp.data.numpy()[0][:3])
        #print("logq", logq.data.numpy()[0][:3])
        
        #print("loss, ", ELBO.data.numpy())
        #print("**************")
        return -ELBO    
    
    
    def get_approxmodel(self, DIM, INPUT_SIZE, nhid):
        return models.MLP_cont(INPUT_SIZE, 2*DIM, nhid, Button.VI_loss_function)
    
    def data_gen(self, ps, ls, N_trials, N_blocks, N_balls):
        
        """
        only ps are there, ls are None
        """
        
        s = 1
        
        last = np.empty(shape = (N_trials*N_blocks, 1))
        m_so_far = np.empty(shape = (N_trials*N_blocks, 1))
        Ns = np.empty(shape = (N_trials*N_blocks, 1))
        
        tvals = np.empty(shape = (N_trials*N_blocks, 1))
        
        pad = np.zeros(shape = (N_trials*N_blocks, 1))
        
        for i, p in enumerate(ps):
            
            vals = np.random.normal(p, s, N_trials)
            prvs = vals.copy()
            prvs[1:] = prvs[:-1]
            prvs[0] = 0
            
            N_b = (1.0*np.arange(N_trials))/N_trials
            
            msf = np.cumsum(vals)/(1.0*np.arange(N_trials) + 1)
            msf[1:] = msf[:-1]
            msf[0] = 0
                
            last[i*N_trials : (i+1)*N_trials, 0] = prvs
            m_so_far[i*N_trials : (i+1)*N_trials, 0] = msf
            Ns[i*N_trials : (i+1)*N_trials, 0] = N_b
            
            tvals[i*N_trials : (i+1)*N_trials, 0] = vals            
            
        
        
        X = np.hstack((last, m_so_far, Ns, pad))
        X = torch.from_numpy(X)
        X = X.type(torch.FloatTensor)
        
        Y = torch.from_numpy(tvals).view(-1,1)
        Y = Y.type(torch.FloatTensor)
        
        
        return(X, Y)
    
    
    
    def assign_PL(self, N_balls, N_blocks, fac):
        
        m0 = 0
        
        v = np.sqrt(fac)
        
        priors = np.random.normal(m0, v, N_blocks)            

        return priors, None
    
        
    
    def get_rationalmodel(self, prior_fac, N_trials):
        
        return models.ButtonRational(prior_fac, N_trials)
    