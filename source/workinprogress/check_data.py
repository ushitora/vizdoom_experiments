#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt


# In[ ]:


f = h5py.File("./demonstration/demodata_figure.hdf5", 'r')


# In[ ]:


list(f['1'])


# In[ ]:


f = 


# In[ ]:


imgs = f['1/states'][:]


# In[ ]:


imgs.shape


# In[ ]:


with h5py.File("./demonstration/imitation_learning_v3/demodata.hdf5", 'r') as f:
    print(list(f.keys()))
    print(list(f['1'].keys()))
    actions = f['1/action'][:]
    ammos = f['1/ammo'][:]
    damagecounts = f['1/damagecount'][:]
    deaths = f['1/damagecount'][:]
    frags = f['1/frag'][:]
    healths = f['1/health'][:]
    posxs = f['1/posx'][:]
    posys = f['1/posy'][:]
    states = f['1/states'][:]


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


for s in imgs[70:80]:
    s_ = np.transpose(s, (1,2,0))
    print(s_.shape)
    f = plt.figure()
    ax = f.add_subplot(1,1,1)
    ax.imshow(s_)


# In[ ]:


s = np.transpose(imgs[70], (1,2,0))
plt.xticks(color="None")
plt.yticks(color="None")
plt.tick_params(length=0)
plt.imshow(s)
plt.savefig("deathmatch_screen.pdf")


# In[ ]:


f.savefig("deathmatch_screen.pdf")


# In[ ]:


plt.plot(range(926), damagecounts)


# In[ ]:


plt.plot(range(926), healths)


# In[ ]:


plt.plot(range(926), frags)


# In[ ]:


plt.plot(range(926), h)


# In[ ]:


plt.plot(posys, posxs)


# In[ ]:


plt.imshow

