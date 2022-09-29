#!/usr/bin/env python
# coding: utf-8

# # Import and Initialization

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# standard numerical library imports
import numpy as np

import matplotlib.pyplot as plt
import pandas
import os
import sklearn

from pfn_utils import *


# In[3]:


# configs
train, val, test = 0.6, 0.3, 0.1
Phi_sizes, F_sizes = (128, 128, 128), (128, 128, 128)
num_epoch = 500
batch_size = 1000


# In[4]:


data_base_dir = "/global/home/users/yifengh3/VAE/vec_data/recon_data"
raw_b_signals = np.load(os.path.join(data_base_dir, "pfn_bsignal_no_ht.npz")) 
raw_hv_signals = np.load(os.path.join(data_base_dir, "pfn_hv_signal_no_ht.npz")) 


# In[5]:


print(list(raw_b_signals.keys()))


# # Original Data

# In[6]:


signal1 = raw_b_signals["data"]
signal2 = raw_hv_signals["data"]


# In[7]:


pfn_original, hist1, original_training_data = train_pfn(signal1, signal2, verbose=2)


# In[8]:


analysis(pfn_original,original_training_data[1][2], original_training_data[1][-1])


# In[9]:


loss, original_roc = pfn_original.evaluate(original_training_data[1][2], batch_size=10000)


# Test of recon data

# In[10]:


log_betas = raw_b_signals["beta"]


# In[11]:


# beta_idx = np.logical_and(log_betas<10, log_betas>-5)
# new_betas = log_betas[beta_idx]
# new_betas = new_betas
# print(new_betas)


# In[12]:


signal_1_recons = raw_b_signals["recon"]
signal_2_recons = raw_hv_signals["recon"]


# In[13]:


# roc_res = []
aucs = []
for signal_1, signal_2 in zip(signal_1_recons,signal_2_recons):
    pfn, hist1, [(X,y), (X_train, X_val, X_test,
     Y_train, Y_val, Y_test), class_weight] = train_pfn(signal_1, signal_2, verbose=2)
    loss, auc = pfn.evaluate(X_test, Y_test)
    print('auc = {}'.format(auc))
    aucs.append(auc)
    np.savetxt('auc_log_v2.csv', np.array(aucs), delimiter=',') 
    


# In[ ]:


# roc_res = []
# aucs = []
# for i, (pfn, hist, data) in enumerate(result):
#     preds = pfn.predict(data[1][2], batch_size=10000)
#     pfn_fp, pfn_tp, threshs = roc_curve(data[1][-1][:,1], preds[:,1])
#     roc_res.append([pfn_fp, pfn_tp, threshs])
#     auc = roc_auc_score(data[1][-1][:,1], preds[:,1])
#     aucs.append(auc)
#     print('reconstruct PFN AUC(beta = {:.2f}):{}'.format(log_betas[i],auc))

# exit()

# In[ ]:


plt.figure(figsize=(10,10))
plt.plot(np.power(10,log_betas),aucs)
plt.semilogx()
plt.ylabel("AUC" ,fontsize=40)
plt.xlabel(r"$\beta$" ,fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig(os.path.join("auc_beta_v3.pdf"), dpi=200)
plt.show()


# In[ ]:


np.savez("auc_v3.npz", beta= log_betas, aucs=aucs, original_roc_info=original_roc)

