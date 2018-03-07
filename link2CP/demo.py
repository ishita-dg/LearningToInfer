import numpy as np
import matplotlib
import matplotlib.pyplot as plt
N = 1000
Nbin = 20
# prior prob of choosing urn 1
priors = {}
# liklihood of drawing color 1 from urn 1
lik_params = {}
# liklihood of data coming from urn 1
likls = {}
# posterior probability of data coming from urn 1
posts = {}
conds = ['inf', 'uninf']


def ret_likl(lik_param, data):
    likls = []
    for d in data:
        likls.append(d*lik_param + (1 - lik_param)*(1-d))
    return np.prod(np.array(likls))
        

def plotall(priors, lik_params, likls, posts, which):
    fig = plt.figure(figsize = (12,6))
    figtr = fig.transFigure.inverted()
    
    count = 1
    for cond in conds:
        ax = plt.subplot(2,4,count)
        ax.hist(priors[cond], bins = Nbin, range = (0.0,1.0))
        ax.set_title("Prior params".format(cond), fontsize = 15)
        count += 1
        
        ax0tr = ax.transData
        xy0 = figtr.transform(ax0tr.transform((0.3,10)))

        ax = plt.subplot(2,4,count)
        ax.hist(lik_params[cond], bins = Nbin, range = (0.0,1.0))
        ax.set_title("Likelihood params".format(cond), fontsize = 15)
        count += 1
        
                
        ax = plt.subplot(2,4,count)
        ax.hist(likls[cond], bins = Nbin, range = (0.0,1.0))
        ax.set_title("Likelihood (norm over H)".format(cond), fontsize = 15)
        count += 1
        

        ax = plt.subplot(2,4,count)
        ax.hist(posts[cond], bins = Nbin, range = (0.0,1.0))
        ax.set_title("Posterior probs queried".format(cond), fontsize = 15)
        count += 1
        ax0tr = ax.transData
        xy1 = figtr.transform(ax0tr.transform((0.3,10)))
        
    fig.tight_layout()
    
    line = matplotlib.lines.Line2D((0.744, 0.744), (0.02, 0.98), transform=fig.transFigure)
    
    fig.lines = line,
    

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


priors['inf'] = np.clip(np.random.beta(fac, fac, N), 0.001, 0.999)
priors['uninf'] = np.clip(np.random.beta(1.0/fac, 1.0/fac, N), 0.001, 0.999)

lik_params['inf'] = np.clip(np.random.beta(1.0/fac, 1.0/fac, N), 0.001, 0.999)
lik_params['uninf'] = np.clip(np.random.beta(fac, fac, N), 0.001, 0.999)

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
    
    
    
plotall(priors, lik_params, likls, posts, which)


#********************************************
#3.2 : Vary only diagnosticity of likelihood + 
#      certainty of hypotheses queried
#********************************************

which = str(3.2)
Ndata = 8
Ndiff = 4

#fac = 2

priors['inf'] = np.clip(np.random.beta(5.0, 5.0, N), 0.001, 0.999)
#priors['uninf'] = np.clip(np.random.beta(5.0, 5.0, N), 0.001, 0.999)
priors['uninf'] = np.clip(np.random.beta(0.5, 0.5, N), 0.001, 0.999)

lik_params['inf'] = np.clip(np.random.beta(5.0, 5.0, N), 0.001, 0.999)
lik_params['uninf'] = np.clip(np.random.beta(5.0, 5.0, N), 0.001, 0.999)


data = []

for cond in conds:
    data = []
    for p, l in zip(priors[cond], lik_params[cond]):
        urn = np.random.binomial(1,p)
        if urn:
            lik = l
        else:
            lik = 1 - l
            
        if cond == 'uninf':
            N1s = int(Ndata/2)
        elif cond == 'inf':
            N1s = int(Ndata/2) + (
                (lik > 0.5)*Ndiff - (lik < 0.5)*Ndiff)/2

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

priors['inf'] = np.clip(np.random.beta(fac, fac, N), 0.001, 0.999)
priors['uninf'] = np.clip(np.random.beta(1.0/fac, 1.0/fac, N), 0.001, 0.999)
#priors['uninf'] = np.clip(np.random.beta(fac, fac, N), 0.001, 0.999)

lik_params['inf'] = np.clip(np.random.beta(0.5, 0.5, N), 0.1, 0.9)
lik_params['uninf'] = np.clip(np.random.beta(5.0, 5.0, N), 0.1, 0.9)


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
            (lik > 0.5)*Ndiff - (lik < 0.5)*Ndiff)/2

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





        
