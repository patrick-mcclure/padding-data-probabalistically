import numpy as np
import torch
import glob
from random import shuffle
import math
import pickle

n_epochs = 200
learning_rate = 1.0

n_components_per_epoch = 5000
data_folder_path = ''

data_file_paths = []

with open('codings/mu.pkl','rb') as f:
    mus = pickle.load(f)
with open('codings/sigma.pkl','rb') as f:
    sigmas = pickle.load(f)
rand_inds = list(range(mus.shape[0])) 
shuffle(rand_inds)
maxes = mus.max(axis=0)
mins = mus.min(axis=0)
zs = np.concatenate([mus,(mus + mus[rand_inds,:])/2.0],axis=0)
n_examples = zs.shape[0]
zs = torch.from_numpy(zs)
zs.requires_grad=True
print(mus.shape)
print(zs.shape)
mus = torch.from_numpy(mus)                           
sigmas = torch.from_numpy(sigmas)
sigmas_squared = torch.square(sigmas)
optimizer = torch.optim.SGD([zs],lr=learning_rate)

#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,n_epochs)
rand_inds = list(range(mus.shape[0]))
for e in range(n_epochs):
    #shuffle(rand_inds)
    
    print('EPOCH: ' + str(e+1) + '/' + str(n_epochs))                       
    max_components_log_prob_sum = 0.0
    for i in range(n_examples):
        log_prob = -torch.square(mus - zs[i])/(2.0*sigmas_squared) - 0.5*math.log(2.0*math.pi) - torch.log(sigmas) 
        log_prob_sum = torch.sum(log_prob,dim=1)
        max_component_log_prob = torch.max(log_prob_sum)
        max_components_log_prob_sum += torch.sum(max_component_log_prob)/n_examples
    print("MAX_COMPONENT_SUM")
    print(max_components_log_prob_sum)
    optimizer.zero_grad()
    max_components_log_prob_sum.backward()
    optimizer.step()
    zs_np = zs.detach().numpy()
    np.save('new_zs/new_zs_'+str(e)+'.npy',np.clip(zs_np,a_min=-6.0,a_max=6.0))