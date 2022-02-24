import numpy as np
import torch
import glob
import random
import math
import pickle 

with open('/Users/mcclureps/Code/codings/mu.pkl','rb') as f: mu = pickle.load(f)
with open('/Users/mcclureps/Code/codings/sigma.pkl','rb') as f: sigma = pickle.load(f)
zs = np.load('/Users/mcclureps/Code/probabalistic-data-padding/new_zs.npy')
print(mu)
print(zs)
#print(sigma)

maxes = mu.max(axis=0)
mins = mu.min(axis=0)
print(maxes)
print(mins)