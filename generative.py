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

       
    
    def get_approxmodel(self, NUM_LABELS, INPUT_SIZE, nhid):
        
        return models.MLP_disc(INPUT_SIZE, NUM_LABELS, nhid, None, Urn.VI_loss_function_grad)
    
    
        #return models.MLP_disc(INPUT_SIZE, 2, nhid, loss_function = nn.KLDivLoss())
        

    
    def data_gen(self, ps, ls, N_trials, N_blocks, N_balls, expt_name = None):
        
        '''
        TODO:
        1. Check dimensionality differences in ls from replications vs NU
        Change -- have NU provide a 2-D input as well
        '''
        
        if expt_name == 'PE': sbsu = True
        else: sbsu = False        
        
    
        draws = np.empty(shape = (N_trials*N_blocks, 1))
        lik1s = np.empty(shape = (N_trials*N_blocks, 1))
        lik2s = np.empty(shape = (N_trials*N_blocks, 1))
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

            if sbsu:
                urn_b = np.random.binomial(1, p, 1)*np.ones(N_trials)
            else:
                urn_b = np.random.binomial(1, p, N_trials)
             

            if expt_name == 'NU':
                '''
                Check with TODO above
                '''
                if l < 0.5 : l = 1.0 - l
            draws_b = []
            delNs = [0]
            delN = 0
            for urn in urn_b:
                if urn:
                    lik = l[1]
                else:
                    lik = l[0]
                # lik will always be the higher prob
                # we want to draw col 0 with this prob
                draw = np.random.binomial(1, lik, 1)
                draws_b.append(draw)
                success = draw*(1-urn) + (1-draw)*urn
                if expt_name == 'PE':
                    delN += (2*success - 1)[0]
                else:
                    delN = 0
                delNs.append(delN)
                
            Ns[i*N_trials : (i+1)*N_trials, 0] = delNs[:-1]                
            draws[i*N_trials : (i+1)*N_trials, 0] = draws_b
            urns[i*N_trials : (i+1)*N_trials, 0] = np.zeros(N_trials)
            lik1s[i*N_trials : (i+1)*N_trials, 0] = l[0] * np.ones(N_trials)
            lik2s[i*N_trials : (i+1)*N_trials, 0] = l[1] * np.ones(N_trials)            
            pris[i*N_trials : (i+1)*N_trials, 0] = p * np.ones(N_trials)
            
        
        X = np.hstack((draws, lik1s, lik2s, pris, Ns))
        X = torch.from_numpy(X)
        X = X.type(torch.FloatTensor)
        
        Y = torch.from_numpy(urns)
        Y = Y.type(torch.LongTensor)
        
        
        return(X, Y)
    
    
    
    def assign_PL_EricUrn(self, N_balls, N_blocks, fac):
        
        # but we don't want all of one color ever
        uninf = [4.0, 5.0, 5.0, 6.0]
        inf = [1.0, 2.0, 2.0, 3.0, 7.0, 8.0, 8.0, 9.0]
        
        if fac < 1.0:
            priors = np.random.choice(inf, N_blocks)/10.0
            likls = np.random.choice(uninf, N_blocks)            
        else:
            priors = np.random.choice(uninf, N_blocks)/10.0
            likls = np.random.choice(inf, N_blocks)
            
        
        return priors, likls
    
    def assign_PL_CP(self, N_balls, blocks, which, alpha_post, alpha_pre):
        
        if which == "test":
            alpha_post = alpha_pre = 1.0
            
        posts = np.random.beta(alpha_post, alpha_post, blocks)
        #post_los = np.log(posts/(1.0 - posts))
        pres = np.random.beta(alpha_pre, alpha_pre, blocks)
        #pre_los = np.log(pres/(1.0 - pres))
        priors = []
        likls = []
        
        for pre, post in zip(pres, posts):
            x = (post*(1.0 - pre))/(pre*(1.0 - post))
            edit = x / (1.0 + x)
            
            ep = np.clip(np.round(edit*N_balls), 1, N_balls - 1)
            pp = np.clip(np.round(pre*N_balls), 1, N_balls - 1)
            if (np.random.uniform() > 0.5):
                # prior is pre
                priors.append(pp*1.0 / N_balls)
                likls.append([ep, N_balls - ep])
            else:
                priors.append(ep*1.0 / N_balls)
                likls.append([pp, N_balls - pp])                
                
        
        return np.array(priors).reshape((-1,1)), np.array(likls).reshape((-1,2))    
    
    def assign_PL_replications(self, N_balls, N_blocks, cond, expt_name, Nlc = None):
        
        if expt_name == "PM":
            Ps = np.linspace(0.1,0.9,9)
            LRs = np.array([[3.0,2.0], [4.0,2.0], [5.0,2.0], [5.0,1.0]])
            #LRs = np.array([[3.0,2.0], [4.0,2.0], [5.0,2.0], [5.0,1.0], [3.0, 3.0], [1.0, 1.0], [5.0, 5.0]])

        elif expt_name == "PE":
            Ps = np.array([0.5])
            LRs = np.array([[85.0, 15.0], [70.0,30.0], [55.0,45.0]])            
            
        if Nlc is not None:
            LRs = LRs[Nlc].reshape(-1,2)
            
        # but we don't want all of one color ever
        cond = 'test'
        
        if cond == 'train':
        
            alpha_p = beta_p = 1.0
            priors = np.round(10*np.random.beta(alpha_p, beta_p, N_blocks))/10
            priors = np.clip(priors, 0.1, 0.9)
            
            alpha_l = beta_l = 1.0
            l1s = np.round(N_balls*np.random.beta(alpha_l, beta_l, N_blocks))
            l1s = np.clip(l1s, 1, N_balls - 1)
            l2s = np.round(N_balls*np.random.beta(alpha_l, beta_l, N_blocks))
            l2s = np.clip(l2s, 1, N_balls - 1)
            
            l_inds = ((2*l1s - 1)/(2*l2s - 1) > 0).astype('int')
            
            likls = np.vstack((l1s, l2s))
            
        elif cond == "test":
            
            priors = np.random.choice(Ps, N_blocks)
            l_inds = np.random.choice(np.arange(len(LRs)), N_blocks)
            which_urn = np.random.choice([1,1], N_blocks)
            
            likls0 = LRs[l_inds]
            likls = np.array([l[::wu] for l,wu in zip(likls0, which_urn)])
        
        return priors.reshape((-1,1)), likls.reshape((-1,2)), l_inds     
    

        
    
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
        
        count = 0
        while count < nsamps:
            s = np.random.normal(qmu, np.exp(qlsd))
            #ELBO = dist_lik.log_prob(s) + dist_pr.log_prob(s)
            if np.isinf(target[1,1]):
                val = gradq(s) * (dist_pr(s) - dist_q(s))
            else:
                val = gradq(s) * (dist_lik(s) + dist_pr(s) - dist_q(s))
            if count == 0 :
                ELBO_grad = val
            else:
                ELBO_grad += val
            count += 1
            
            
        
        #print(torch.mean(torch.cat(ELBOs)).data.numpy(), qentropy.data.numpy(), "***")
        
        grad = ELBO_grad/count
        #print(grad)
    
        return autograd.Variable(torch.Tensor(grad).type(torch.FloatTensor).view(1,-1)   )
    
    
    def get_approxmodel(self, DIM, INPUT_SIZE, nhid):
        return models.MLP_cont(INPUT_SIZE, DIM, nhid, None, Button.VI_loss_function_grad)
    
    def data_gen(self, ps, lik_var, N_trials):
        '''
        TODO: revise
        '''
        
        lik_sd = np.sqrt(lik_var)
        N_blocks = len(ps)
        
        last = np.empty(shape = (N_trials*N_blocks, 1))
        m_so_far = np.empty(shape = (N_trials*N_blocks, 1))
        Ns = np.empty(shape = (N_trials*N_blocks, 1))
        
        tvals = np.empty(shape = (N_trials*N_blocks, 1))
        for i, p in enumerate(ps):
            
            vals = np.random.normal(p, lik_sd, N_trials)
            prvs = vals.copy()
            prvs[1:] = prvs[:-1]
            prvs[0] = 0
            
            N_b = (1.0*np.arange(N_trials))
            #N_b = (1.0*(np.arange(N_trials) + 1.0))/N_trials
            
            msf = np.cumsum(vals)/(N_b + 1.0)
            msf[1:] = msf[:-1]
            msf[0] = 0
                
            last[i*N_trials : (i+1)*N_trials, 0] = prvs
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
    