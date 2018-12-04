
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np


# In[ ]:


WEIGHT_PREDICT_RISKSUICIDE_DIR = "./weights_suicide/"
WEIGHT_PREDICT_ENEMYPOS_DIR = "./weights_enemypos/"


# In[ ]:


weights_suicide = np.load(WEIGHT_PREDICT_RISKSUICIDE_DIR+"conv1_kernel.npy")
weights_enemy = np.load(WEIGHT_PREDICT_ENEMYPOS_DIR+"conv1_kernel.npy")
weights_new = np.zeros_like(weights_suicide)
weights_new[:,:,:,:16] = weights_suicide[:,:,:,:16]
weights_new[:,:,:,16:] = weights_enemy[:,:,:,:16]
np.save("./weights_merged/conv1_kernel.npy",weights_new)


# In[ ]:


weights_suicide = np.load(WEIGHT_PREDICT_RISKSUICIDE_DIR+"conv1_bias.npy")
weights_enemy = np.load(WEIGHT_PREDICT_ENEMYPOS_DIR+"conv1_bias.npy")
weights_new = np.zeros_like(weights_suicide)
weights_new[:16] = weights_suicide[:16]
weights_new[16:] = weights_enemy[:16]
np.save("./weights_merged/conv1_bias.npy",weights_new)


# In[ ]:


weights_suicide = np.load(WEIGHT_PREDICT_RISKSUICIDE_DIR+"conv2_kernel.npy")
weights_enemy = np.load(WEIGHT_PREDICT_ENEMYPOS_DIR+"conv2_kernel.npy")
weights_new = np.zeros_like(weights_suicide)
weights_new[:,:,:,:16] = weights_suicide[:,:,:,:16]
weights_new[:,:,:,16:] = weights_enemy[:,:,:,:16]
np.save("./weights_merged/conv2_kernel.npy",weights_new)


# In[ ]:


weights_suicide = np.load(WEIGHT_PREDICT_RISKSUICIDE_DIR+"conv2_bias.npy")
weights_enemy = np.load(WEIGHT_PREDICT_ENEMYPOS_DIR+"conv2_bias.npy")
weights_new = np.zeros_like(weights_suicide)
weights_new[:16] = weights_suicide[:16]
weights_new[16:] = weights_enemy[:16]
np.save("./weights_merged/conv2_bias.npy",weights_new)

