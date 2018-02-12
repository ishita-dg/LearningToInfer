import autograd.numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import json

def gaussian_logpdf(yval, samples):
    #mean, std = yval[0,:D].view(1, -1), torch.exp(yval[0,-D:].view(1, -1))
    mean, std = yval[0,0], torch.exp(yval[0,1])
    std = 1.0 * std/std
    lprob = -(((samples - mean)/std)**2 + torch.log(2*np.pi*std**2))/2
    return lprob.view(1, -1)


def gaussian_entropy(std):
    log_std = torch.log(std)
    norm = autograd.Variable(torch.Tensor([2*np.pi]))
    return 0.5 * len(std) * (1.0 + torch.log(norm)) + torch.sum(log_std)

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

def logit(p):
    return np.log(p) - np.log(1 - p)
        
def plot_both(data, model, dset, expt, fac, N_epoch, show = False):
    
    count = 0
    
    if expt == 'disc' :
    
        fig = plt.figure(figsize=(16,8))
        for cond in data:
            count += 1
            ax = fig.add_subplot(1, 2, count)
            ax.plot(2*data[cond][dset]["X"].numpy()[:, 0] - 1, label = "likl/ball_drawn")
            ax.plot(2*data[cond][dset]["X"].numpy()[:, 2] - 1, label = "pri")
            ax.plot(data[cond][dset]["y_pred_" + model].numpy()[:, 1] - 0.5, label = "prediction")
            ax.plot(data[cond][dset]["y"].numpy() - 0.5, label = "true", linestyle = "--")
            ax.set_title('{0}'.format(cond))
            ax.set_ylim([-1.4, 1.4])
            ax.legend()
            
        plt.savefig('figs/{4}_{0}{1}_fac{2}epochs{3}.png'.format(model, dset,round(fac), N_epoch, expt))
    
    elif expt == 'cont':
        
        fig = plt.figure(figsize=(16,8))
        for cond in data:
            count += 1
            ax = fig.add_subplot(1, 2, count)
            ax.plot(data[cond][dset]["X"].numpy()[:, 0], label = "likl/last_val")
            ax.plot(data[cond][dset]["X"].numpy()[:, 1], label = "pri/avg so far")
            ax.plot(data[cond][dset]["y_pred_" + model].numpy()[:, 0], label = "prediction")
            ax.set_title('{0}'.format(cond))
            #ax.set_ylim([-1.4, 1.4])
            ax.legend()
            
        plt.savefig('figs/{4}_{0}{1}_fac{2}epochs{3}.png'.format(model, dset,round(fac), N_epoch, expt))
        if (dset == 'test' and model == 'am'): plt.show()
        
    return
        
        
def updates(array, N_trials, expt, prob = False):
    #prob = True
    if expt == "disc" : prob = True
    if prob:
        ret = np.array([inv_logit(array[i+1]) - inv_logit(array[i]) for i in np.arange(array.size - 1) \
                        if (i%N_trials != 0 and i%N_trials != N_trials-1)])
    else:
        ret = np.array([array[i+1] - array[i] for i in np.arange(array.size - 1) \
                        if (i%N_trials != 0 and i%N_trials != N_trials-1)])
    return ret



def get_binned(fbin, tbin, lim):
    #lim = 0.8
    num = 12
    bins = np.linspace(0,lim,num = num) + np.random.uniform(-0.05, 0.05)
    ind = np.digitize(fbin, bins = bins)
    y = []
    se = []
    x = []
    
    for i in np.arange(num):
        i += 1
        rvals = tbin[ind == i]
        if rvals.size:
            x.append(bins[i-1])
            y.append(np.mean(rvals))
            se.append(np.std(rvals))
    
    return (x, y, se)

def plot_calibration(di, du, N_epoch, sg_epoch, fac, N_blocks, N_trials, expt):
    
    if expt == 'disc':
        inf_hrm = np.abs(inv_logit(di["y_pred_hrm"].numpy()[:,1].flatten()) - di["ps"])
        inf_am = np.abs(inv_logit(di["y_pred_am"].numpy()[:,1].flatten()) - di['ps']) 
    
        uninf_hrm = np.abs(inv_logit(du["y_pred_hrm"].numpy()[:,1].flatten()) - du['ps']) 
        uninf_am = np.abs(inv_logit(du["y_pred_am"].numpy()[:,1].flatten()) - du['ps']) 

    elif expt == 'cont':
        
        inf_hrm = np.abs(di["y_pred_hrm"].numpy()[:,0].flatten())
        inf_am = np.abs(di["y_pred_am"].numpy()[:,0].flatten()) 
        
        uninf_hrm = np.abs(du["y_pred_hrm"].numpy()[:,0].flatten()) 
        uninf_am = np.abs(du["y_pred_am"].numpy()[:,0].flatten()) 
        

        #inf_hrm = np.abs(di["y_pred_hrm"].numpy()[:,0].flatten() - di["ps"])
        #inf_am = np.abs(di["y_pred_am"].numpy()[:,0].flatten() - di['ps']) 
        
        #uninf_hrm = np.abs(du["y_pred_hrm"].numpy()[:,0].flatten() - du['ps']) 
        #uninf_am = np.abs(du["y_pred_am"].numpy()[:,0].flatten() - du['ps']) 
        

    #plt.scatter(inf_hrm, inf_am, alpha =0.1)
    #plt.scatter(uninf_hrm, uninf_am, alpha = 0.1)
    
    lim = max(np.concatenate((inf_hrm, uninf_hrm)))
    if expt == 'disc':
        lim = 1.0
        
    
    ix, iy, ise = get_binned(fbin = inf_hrm, tbin = inf_am, lim = lim)
    ux, uy, use = get_binned(fbin = uninf_hrm, tbin = uninf_am, lim = lim)

    plt.errorbar(ix, iy, ise, label = "low_dispersion")
    plt.errorbar(ux, uy, use, label = "high_dispersion")

    plt.plot([0,lim], [0,lim], c = 'k')
    
    #plt.ylim(0, lim)
    

    plt.legend()   
    plt.title("Calibration for updates")
    plt.xlabel("rational model update")
    plt.ylabel("approx model update")
    plt.legend()
    plt.show()
    
    plt.savefig('figs/updates_{5}_epoch{0}_sg{1}_f{2}_Nb{3}_Nt{4}.png'.format(N_epoch, sg_epoch, fac, N_blocks, N_trials, expt))
    
    
    
    fn = 'data/preds_{5}_epoch{0}_sg{1}_f{2}_Nb{3}_Nt{4}.json'.format(N_epoch, sg_epoch, fac, N_blocks, N_trials, expt)
    data = {'inf_rm_update': list(inf_hrm),
            'uninf_rm_update': list(uninf_hrm),
            'inf_am_update': list(inf_am),
            'uninf_am_update': list(uninf_am),
            'inf_prior' : list(di['ps']),
            'uninf_prior' : list(du['ps'])
    }
    
    #with open(fn, 'wb') as outfile:
        #json.dump(data, outfile)

    
    
def plot_isocontours(ax, func, xlimits=[-20, 20], ylimits=[-20, 20], numticks=101):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
    Z = zs.reshape(X.shape)
    plt.contour(X, Y, Z, 30)
    ax.set_yticks([])
    ax.set_xticks([])

    
def plot_1D(ax, func, xlimits=[-20, 20], numticks=101):
    X = np.linspace(*xlimits, num=numticks)
    Y = func(X)
    plt.plot(X, Y)
    ax.set_yticks([])
    ax.set_xticks([])

    