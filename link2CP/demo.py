import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json

from scipy.stats import gaussian_kde

N = 2000
Nbin = 20
# prior prob of choosing urn 1
priors = {}
# liklihood of drawing color 1 from urn 1
lik_params = {}
# liklihood of data coming from urn 1
likls = {}
# posterior probability of data coming from urn 1
posts = {}
conds = ['inf_lik', 'uninf_lik']


def ret_likl(lik_param, data):
    likls = []
    for d in data:
        likls.append(d*lik_param + (1 - lik_param)*(1-d))
    return np.prod(np.array(likls))
        
def log_odds_func(x):
    return(np.log(x / (1.0 - x)))
    
def plotall(priors, lik_params, likls, posts, which, kde = False, log_odds = False):
    
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off    


    fig = plt.figure(figsize = (12,6))
    figtr = fig.transFigure.inverted()
    
    sig_pr = {'inf_lik' : 0.7,
              'uninf_lik' : 0.25}
    sig_lik = {'inf_lik' : 0.7,
              'uninf_lik' : 0.7}
    sig_post = {'inf_lik' : 0.25,
              'uninf_lik' : 0.25}
    
    Ndisc = 1000
    if log_odds:
        f = log_odds_func
        g_range = (-4, 4)
        Xlab = "Log odds"
    else:
        f = lambda x : x
        g_range = (0,1)
        Xlab = "Probabilities"
    
    delx = (g_range[1] - g_range[0])/float(Ndisc)
    count = 1
    
    for cond in conds:
        ax = plt.subplot(2,4,count)
        if kde:
            density = gaussian_kde(f(priors[cond]))
            density.covariance_factor = lambda : sig_pr[cond]
            density._compute_covariance()        
            x = np.linspace(g_range[0], g_range[1], Ndisc)
            ax.plot(x, density(x) / sum(delx*density(x)))
            ax.set_ylim([0,1.2])
        else:
            ax.hist(f(priors[cond]), bins = Nbin, normed = True, range = g_range)
        ax.set_xlabel(Xlab)
        ax.set_ylabel("Probability density")
        ax.set_title("Prior".format(cond), fontsize = 15)
        #ax.set_yticks([])
        count += 1
        
        ax0tr = ax.transData
        xy0 = figtr.transform(ax0tr.transform((0.25,10)))

        ax = plt.subplot(2,4,count)
        if kde:
            density = gaussian_kde(f(lik_params[cond]))
            density.covariance_factor = lambda : sig_lik[cond]
            density._compute_covariance()        
            x = np.linspace(g_range[0], g_range[1], Ndisc)
            ax.plot(x, density(x)/ sum(delx*density(x)))
            ax.set_ylim([0,1.2])
        else:
            ax.hist(f(lik_params[cond]), bins = Nbin, normed = True, range = g_range)
        ax.set_xlabel(Xlab)
        ax.set_ylabel("Probability density")
        ax.set_title("Likelihood".format(cond), fontsize = 15)
        #ax.set_yticks([])
        count += 1
        
                
        if not log_odds:
            ax = plt.subplot(2,4,count)
            #ax.hist(f(likls[cond]), bins = Nbin, normed = True, range = g_range)
            ax.set_xlabel(Xlab)
            ax.set_ylabel("Probability density")
            ax.set_title("Likelihood (norm over H)".format(cond), fontsize = 15)
            count += 1
        

        ax = plt.subplot(2,4,count)
        if kde:
            density = gaussian_kde(f(posts[cond]))
            density.covariance_factor = lambda : sig_post[cond]
            density._compute_covariance()        
            x = np.linspace(g_range[0], g_range[1], Ndisc)
            ax.plot(x, density(x)/ sum(delx*density(x)))
            ax.set_ylim([0,1.2])
        else:
            ax.hist(f(posts[cond]), bins = Nbin, normed = True, range = g_range)
        ax.set_xlabel(Xlab)
        ax.set_ylabel("Probability density")
        ax.set_title("Posterior".format(cond), fontsize = 15)
        #ax.set_yticks([])
        count += 1
        ax0tr = ax.transData
        xy1 = figtr.transform(ax0tr.transform((0.25,10)))
        
        if log_odds: count +=1
        
    fig.tight_layout()
    
    line = matplotlib.lines.Line2D((0.744, 0.744), (0.02, 0.98), transform=fig.transFigure)
    
    if not log_odds: fig.lines = line,
    

    plt.savefig("type_{0}.png".format(which))
    #plt.show()
    return

#********************************************    
# 3.1 : Vary only parameters + diagnosticity 
#       of likelihood (LTI)
#********************************************

which = str(3.1)
Ndata = 1
fac = 10.0
mid = [0.5, 0.5, 0.4, 0.6]
end = [0.1,0.2,0.2,0.3,0.7,0.8,0.8,0.9]

priors['inf_lik'] = np.random.choice(mid, N)
priors['uninf_lik'] = np.random.choice(end, N)

lik_params['inf_lik'] = np.random.choice(end, N)
lik_params['uninf_lik'] = np.random.choice(mid, N)


#priors['inf_lik'] = np.clip(np.random.beta(fac, fac, N), 0.001, 0.999)
#priors['uninf_lik'] = np.clip(np.random.beta(1.0/fac, 1.0/fac, N), 0.001, 0.999)

#lik_params['inf_lik'] = np.clip(np.random.beta(1.0/fac, 1.0/fac, N), 0.001, 0.999)
#lik_params['uninf_lik'] = np.clip(np.random.beta(fac, fac, N), 0.001, 0.999)

data = []

for cond in conds:
    data = []
    for p, l in zip(priors[cond], lik_params[cond]):
        urn = np.random.binomial(1,p)
        if urn:
            lik = l
        else:
            lik = 1 - l
        data.append(1 - np.random.binomial(1, lik, Ndata))
        
    data = np.array(data)
            
    likls[cond] = np.array([ret_likl(lp, d) for lp, d in zip(lik_params[cond], data)])
    
    likls_other = np.array([ret_likl(lp, d) for lp, d in zip(1 - lik_params[cond], data)])
    
    posts[cond] = likls[cond] * priors[cond] / (likls[cond] * priors[cond] + likls_other * (1 - priors[cond]))

    likls[cond] = likls[cond] / (likls[cond] + likls_other)
    
    
    
plotall(priors, lik_params, likls, posts, which)#, log_odds = True)
#Store log odds
if N == 100:
    for name, t in zip(('priors', 'likls', 'posts'), (priors, lik_params, posts)):
        for key in t:
            t[key] = list(log_odds_func(t[key]))
            
        with open(name + '.txt', 'w') as fn:
            json.dump(t,fn)
            
        
#********************************************
#3.2 : Vary only diagnosticity of likelihood + 
#      certainty of hypotheses queried
#********************************************

which = str(3.2)
Ndata = 8
Ndiff = 4

#fac = 2

priors['inf_lik'] = np.clip(np.random.beta(5.0, 5.0, N), 0.001, 0.999)
#priors['uninf_lik'] = np.clip(np.random.beta(5.0, 5.0, N), 0.001, 0.999)
priors['uninf_lik'] = np.clip(np.random.beta(0.25, 0.25, N), 0.001, 0.999)

lik_params['inf_lik'] = np.clip(np.random.beta(5.0, 5.0, N), 0.001, 0.999)
lik_params['uninf_lik'] = np.clip(np.random.beta(5.0, 5.0, N), 0.001, 0.999)


data = []

for cond in conds:
    data = []
    for p, l in zip(priors[cond], lik_params[cond]):
        urn = np.random.binomial(1,p)
        if urn:
            lik = l
        else:
            lik = 1 - l
            
        if cond == 'uninf_lik':
            N1s = int(Ndata/2)
        elif cond == 'inf_lik':
            N1s = int(Ndata/2) + (
                (lik > 0.25)*Ndiff - (lik < 0.25)*Ndiff)/2

        N0s = Ndata - N1s
        

        manipulated_data = np.array([0]*N0s + [1]*N1s)
        np.random.shuffle(manipulated_data)
        data.append(manipulated_data)
        
        
    data = np.array(data)
            
    likls[cond] = np.array([ret_likl(lp, d) for lp, d in zip(lik_params[cond], data)])
    
    likls_other = np.array([ret_likl(lp, d) for lp, d in zip(1 - lik_params[cond], data)])
    posts[cond] = likls[cond] * priors[cond] / (likls[cond] * priors[cond] + likls_other * (1 - priors[cond]))
    
    likls[cond] = likls[cond] / (likls[cond] + likls_other)
    
    
plotall(priors, lik_params, likls, posts, which)

#********************************************
#3.3 : Vary only parameters OR only parameters 
#      + certainty of hypotheses queried
#********************************************


which = str(3.3)
Ndata = 10
odds = 3

fac = 10

priors['inf_lik'] = np.clip(np.random.beta(fac, fac, N), 0.001, 0.999)
priors['uninf_lik'] = np.clip(np.random.beta(1.0/fac, 1.0/fac, N), 0.001, 0.999)
#priors['uninf_lik'] = np.clip(np.random.beta(fac, fac, N), 0.001, 0.999)

lik_params['inf_lik'] = np.clip(np.random.beta(0.25, 0.25, N), 0.1, 0.9)
lik_params['uninf_lik'] = np.clip(np.random.beta(5.0, 5.0, N), 0.1, 0.9)


data = []

for cond in conds:
    data = []
    for p, l in zip(priors[cond], lik_params[cond]):
        urn = np.random.binomial(1,p)
        if urn:
            lik = l
        else:
            lik = 1 - l
            
        r = lik/(1.0*(1-lik))
        Ndiff = int(np.round(np.log(odds)/np.log(r)))
        N1s = int(Ndata/2) + (
            (lik > 0.25)*Ndiff - (lik < 0.25)*Ndiff)/2

        N0s = Ndata - N1s
        
        #if not Ndiff:
            #print(lik, np.log(r))
        

        manipulated_data = np.array([0]*N0s + [1]*N1s)
        np.random.shuffle(manipulated_data)
        data.append(manipulated_data)
        
        
    data = np.array(data)
            
    likls[cond] = np.array([ret_likl(lp, d) for lp, d in zip(lik_params[cond], data)])
    
    likls_other = np.array([ret_likl(lp, d) for lp, d in zip(1 - lik_params[cond], data)])
    posts[cond] = likls[cond] * priors[cond] / (likls[cond] * priors[cond] + likls_other * (1 - priors[cond]))
    
    likls[cond] = likls[cond] / (likls[cond] + likls_other)
    
    
plotall(priors, lik_params, likls, posts, which)





        
