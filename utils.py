import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

def logit(p):
    return np.log(p) - np.log(1 - p)
        
def plot_both(data, model, dset, expt, fac, N_epoch):
    
    count = 0
    
    if expt == 'disc' :
    
        fig = plt.figure(figsize=(16,8))
        for cond in data:
            count += 1
            ax = fig.add_subplot(1, 2, count)
            ax.plot(2*data[cond][dset]["X"].numpy()[:, 0] - 1, label = "likl/ball_drawn")
            ax.plot(data[cond][dset]["X"].numpy()[:, 2], label = "pri/N_left")
            ax.plot(inv_logit(data[cond][dset]["y_pred_" + model].numpy()) - 0.5, label = "prediction")
            #print("\n", model, data[cond][dset]["y_pred" + model] - 0.5)
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
            ax.plot(data[cond][dset]["X"].numpy()[:, 3], label = "pri/avg so far")
            ax.plot(data[cond][dset]["y_pred_" + model].numpy(), label = "prediction")
            #print("\n", model, data[cond][dset]["y_pred" + model] - 0.5)
            #ax.plot(data[cond][dset]["y"].numpy(), label = "true", linestyle = "--")
            ax.set_title('{0}'.format(cond))
            #ax.set_ylim([-1.4, 1.4])
            ax.legend()
            
        plt.savefig('figs/{4}_{0}{1}_fac{2}epochs{3}.png'.format(model, dset,round(fac), N_epoch, expt))
        
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
    num = 10
    bins = np.linspace(0,lim,num = num) + np.random.uniform(-0.05, 0.05)
    ind = np.digitize(fbin, bins = bins)
    y = []
    se = []
    x = []
    
    for i in np.arange(50):
        i += 1
        rvals = tbin[ind == i]
        if rvals.size:
            x.append(bins[i-1])
            y.append(np.mean(rvals))
            se.append(np.std(rvals))
    
    return (x, y, se)

def plot_calibration(di, du, N_epoch, sg_epoch, fac, N_blocks, N_trials, expt):
    
    inf_hrm = np.abs(updates(di["y_pred_hrm"].numpy().flatten(), N_trials, expt))
    inf_am = np.abs(updates(di["y_pred_am"].numpy().flatten(), N_trials, expt) )
    
    uninf_hrm = np.abs(updates(du["y_pred_hrm"].numpy().flatten(), N_trials, expt))
    uninf_am = np.abs(updates(du["y_pred_am"].numpy().flatten(), N_trials, expt))
    
    plt.scatter(inf_hrm, inf_am, alpha =0.1)
    plt.scatter(uninf_hrm, uninf_am, alpha = 0.1)
    
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