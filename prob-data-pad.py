import numpy as np
import torch
import glob
import random

n_classes = 7

n_sample = 10

data_folder_path = ''

data_file_paths = []

for n in range(n_classes):
  data_file_paths.append(glob.glob(data_folder_path + '/' + str(n))

z_m_list = []
z_s_list = [] 

i = 0
for n in range(n_classes):
  random.shuffle(data_file_paths[n])
  for k in range(n_samples):
    tmp_array = np.load(data_file_paths[n])
    z_m_list.append(torch.from_numpy(tmp_array[0]))
    z_s_list.append(torch.from_numpy(tmp_array[1]))
 z_m = torch.stack(z_m_list)
 shuffle(z_m_list)
 z_m_shuffled = torch.stack(z_m_list)
 z_s = torch.stack(z_s_list)
 
 z_new = (z_m + z_m_shuffled)/2.0
 
