import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
matplotlib.rc('xtick', labelsize=10) 
matplotlib.rc('ytick', labelsize=10) 
plt.rcParams['axes.titlepad'] = 20 
import numpy as np
from scipy.stats import multivariate_normal
import scipy.ndimage as ndimage

def find_KL(P, Q):
    val = P*np.log(P/Q) + (1.0 - P)*np.log((1.0 - P)/(1.0 - Q))
    return val

def find_post(pr, lik):
    post = np.array([pr*lik, (1.0 - pr)*(1.0 - lik)])
    return post[1]/ np.sum(post)

cmap = 'RdYlGn'

gap = 0.01
probs = np.arange(gap, 1.0, gap)
N = len(probs)
posts = np.empty((N, N))
for i, pr in enumerate(probs):
    for j, lik in enumerate(probs):
        posts[i, N - 1 - j] = find_post(pr, lik)
plt.imshow(posts, cmap=cmap, interpolation='nearest')
plt.clim(0.0, 1.0)
#plt.colorbar()

x = np.arange(0, N)
y = np.arange(0, N)
X, Y = np.meshgrid(x, y)

var0 = multivariate_normal(mean=[75, 35], cov=[[120,20],[20,120]])
var1 = multivariate_normal(mean=[30, 35], cov=[[170,-100],[-100,170]])
Z = np.empty((N,N))
for i, x0 in enumerate(x):
    for j, y0 in enumerate(y):
        Z[i, j] =  var0.pdf([x0, y0]) + var1.pdf([x0, y0])
        
plt.contour(X, Y, Z, colors = ('k',))

factor = 25.0
plt.xticks(np.clip(np.arange(0.0, 100.0 + factor, factor), np.min(X), np.max(X)), np.arange(0.0/100.0, 1.0+ factor/100.0, factor/100.0))
plt.yticks(np.clip(np.arange(0.0, 100.0 + factor, factor), np.min(X), np.max(X)), 1.0 - np.arange(0.0/100.0, 1.0 + factor/100.0, factor/100.0))
plt.xlabel("Prior")
plt.ylabel("Likelihood")
#plt.title("True posterior P")

#plt.savefig('figs/all_posts.png')


m1, m2, m3, c = np.random.uniform(-3, 3, 4)
def find_random(pr, lik):
    y = m1*pr + m2*lik + m3*(1-pr)*lik + c
    y = np.random.normal(0.0, 4.0)
    return 1.0/ (1.0 + np.exp(y))

gap = 0.01
probs = np.arange(gap, 1.0, gap)
N = len(probs)
Qs = np.empty((N, N))
for i, pr in enumerate(probs):
    for j, lik in enumerate(probs):
        Qs[i, j] = find_random(pr, lik)
plt.imshow(Qs, cmap=cmap, interpolation='nearest')
plt.clim(0.0, 1.0)
plt.contour(X, Y, Z, colors = ('k',))

plt.xticks(np.clip(np.arange(0.0, 100.0 + factor, factor), np.min(X), np.max(X)), np.arange(0.0/100.0, 1.0+ factor/100.0, factor/100.0))
plt.yticks(np.clip(np.arange(0.0, 100.0 + factor, factor), np.min(X), np.max(X)), 1.0 - np.arange(0.0/100.0, 1.0 + factor/100.0, factor/100.0))

#plt.title("Estimated posterior parameter P")
#plt.savefig('figs/all_Qs.png')


maxZ = np.max(Z)
Z_scaled = Z/maxZ
v_high = 0.08

final_Qs = Qs
final_Qs[Z_scaled > v_high] = posts[Z_scaled > v_high] 
final_Qs_smooth = ndimage.gaussian_filter(final_Qs, sigma=(5), order=0)
plt.imshow(final_Qs_smooth, cmap=cmap, interpolation='nearest')
#plt.colorbar()
plt.clim(0.0, 1.0)
plt.contour(X, Y, Z, colors = ('k',))

plt.xticks(np.clip(np.arange(0.0, 100.0 + factor, factor), np.min(X), np.max(X)), np.arange(0.0/100.0, 1.0+ factor/100.0, factor/100.0))
plt.yticks(np.clip(np.arange(0.0, 100.0 + factor, factor), np.min(X), np.max(X)), 1.0 - np.arange(0.0/100.0, 1.0 + factor/100.0, factor/100.0))



#plt.title("Estimated posterior Q")
#plt.savefig('figs/all_finalQs.png')

KLs = find_KL(final_Qs, posts)
KLs_smooth = ndimage.gaussian_filter(KLs, sigma=(5), order=0)
plt.imshow(KLs_smooth, cmap='PuBu', interpolation='nearest')
plt.colorbar()
plt.clim(0, 0.6)
plt.contour(X, Y, Z, colors = ('k',))

plt.xticks(np.clip(np.arange(0.0, 100.0 + factor, factor), np.min(X), np.max(X)), np.arange(0.0/100.0, 1.0+ factor/100.0, factor/100.0))
plt.yticks(np.clip(np.arange(0.0, 100.0 + factor, factor), np.min(X), np.max(X)), 1.0 - np.arange(0.0/100.0, 1.0 + factor/100.0, factor/100.0))


#plt.title("KL (P || Q)")
plt.savefig('figs/all_KLs.png')




