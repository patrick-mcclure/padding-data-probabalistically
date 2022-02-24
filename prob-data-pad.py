import numpy as np
import torch
import glob
from random import shuffle
import math
import pickle

n_epochs = 200
learning_rate = 0.1
n_examples = 1000
n_components_per_epoch = 5000
data_folder_path = ''

data_file_paths = []

with open('codings/mu.pkl','rb') as f:
    mus = pickle.load(f)
with open('codings/sigma.pkl','rb') as f:
    sigmas = pickle.load(f)
rand_inds = list(range(mus.shape[0]))                           
maxes = mus.max(axis=0)
mins = mus.min(axis=0)
zs = np.zeros([n_examples,mus.shape[1]])
for i in range(mus.shape[1]):
    zs[:,i] = np.random.uniform(low = mins[i], high = maxes[i], size = n_examples) 
#zs = (mus + mus[rand_inds,:])/2.0

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
        #prob = torch.exp(-torch.square(mus - zs[i])/(2.0*sigmas_squared))/(math.sqrt(2.0*math.pi)*sigmas)
        #print("LOG_PROB")
        #print(log_prob)
        log_prob_sum = torch.sum(log_prob,dim=1)
        #prob_prod = torch.prod(prob,dim=1)
        #print("LOG_PROB_SUM")
        #print(log_prob_sum)
        max_pool = torch.nn.MaxPool1d(mus.shape[0])
        max_component_log_prob = max_pool(torch.reshape(log_prob_sum,[1,1,-1]))
        #print("MAX_COMPONENT")
        #print(max_component_log_prob)
        max_components_log_prob_sum += torch.sum(max_component_log_prob)/n_examples
        #max_components_log_prob_sum += torch.sum(prob_prod)/(n_examples*mus.shape[0])
    print("MAX_COMPONENT_SUM")
    print(max_components_log_prob_sum)
    optimizer.zero_grad()
    max_components_log_prob_sum.backward()
    optimizer.step()
    #scheduler.step()
    zs_np = zs.detach().numpy()
    #for i in range(mus.shape[1]):
        #zs_np[:,i] = np.clip(zs_np[:,i],mins[i],maxes[i])
    #print(zs_np)

    
np.save('new_zs.npy',zs_np)
 