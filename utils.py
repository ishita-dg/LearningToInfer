import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

        
def plot_both(data, model, dset, fac, N_epoch):
    
    count = 0
    fig = plt.figure(figsize=(16,8))
    for cond in data:
        count += 1
        ax = fig.add_subplot(1, 2, count)
        ax.plot(2*data[cond][dset]["X"].numpy()[:, 0] - 1, label = "likl/ball_drawn")
        ax.plot(data[cond][dset]["X"].numpy()[:, 2], label = "pri/N_left")
        ax.plot(data[cond][dset]["y_pred" + model] - 0.5, label = "prediction")
        #print("\n", model, data[cond][dset]["y_pred" + model] - 0.5)
        ax.plot(data[cond][dset]["y"].numpy() - 0.5, label = "true", linestyle = "--")
        ax.set_title('{0}'.format(cond))
        ax.set_ylim([-1.4, 1.4])
        ax.legend()
        
        plt.savefig('figs/{0}{1}_fac{2}epochs{3}.png'.format(model, dset,round(fac), N_epoch))
        
        
def updates(array, prob = False):
    if prob:
        ret = np.array([np.log(array[i+2]) - np.log(array[i+1]) for i in np.arange(array.size - 2)])
    else:
        ret = np.array([array[i+2] - array[i+1] for i in np.arange(array.size - 2)])
    return ret