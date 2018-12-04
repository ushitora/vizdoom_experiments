
# coding: utf-8

# In[ ]:


import numpy as np
from vizdoom import *
import skimage.color, skimage.transform
from sklearn.model_selection import train_test_split
from random import sample, randint, random
import time,random,threading,datetime
from tqdm import tqdm
import tensorflow as tf
import h5py
import sys, os, glob
from PIL import Image
import matplotlib.pyplot as plt
from game_instance import GameInstance
from global_constants import *


# In[ ]:


__name__ = "learning"

DEMO_PATH = ["./demonstration/demodata"+"%02d"%(i)+".hdf5" for i in range(1,3)]

# RESOLUTION = (120,160,3)
# RESOLUTION = (90, 120, 3)

THRESHOLD = 1000
KEEP_PROB = 0.5


# In[ ]:


def convert_action_engine2agent(engine_action):
        assert type(engine_action) == type(list()), print("type: ", type(engine_action))

        ans = 0
        for i, e_a in enumerate(engine_action):
            ans += e_a * 2**i

        return ans


# In[ ]:


def load_data():
    batch_img = []
    batch_label = []
    for d in DEMO_PATH:
        print("loading "+d)
        file = h5py.File(d, "r")
        episode_list = list(file.keys())[1:]

        for e in episode_list[:]:
            n_steps = file[e+"/states"].shape[0]
            states = file[e+"/states"][:]
            actions = file[e+"/action"][:]
            buff = np.zeros(shape=(N_ADV,)+RESOLUTION)
            for i,img in enumerate(states):
                buff[i%N_ADV] = img
                if i%N_ADV==N_ADV-1 or i==n_steps-1:
                    batch_img.append(np.copy(buff))
                    buff =  np.zeros(shape=(N_ADV,)+RESOLUTION)

            buff = []
            for i,act in enumerate(actions):
                buff.append(convert_action_engine2agent(act.tolist()))

                if i%N_ADV == N_ADV -1or i==n_steps-1:
                    batch_label.append(np.copy(buff))
                    buff = []

        file.close()
    return np.array(batch_img), np.array(batch_label)


# In[ ]:


class NetworkLocal(object):
    def __init__(self,name):
        self.name = name
        
        self.state1_ = tf.placeholder(tf.float32,shape=(None,N_ADV,)+RESOLUTION, name="state_1")
        self.target_ = tf.placeholder(tf.int32, shape=(None,), name="action")
        self.keep_prob_ = tf.placeholder(tf.float32)
        
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.q_model = self._model(self.state1_, self.keep_prob_)
            
        self.saver = tf.train.Saver()

        self._build_graph()

#         print("-----LOCAL weights---")
#         for w in self.weights_params:
#             print(w)
            
#         print("-----LOCAL grads---")
#         for w in self.grads:
#             print(w)
    
    def _model(self,state, keep_prob):

        self.conv1 = NetworkSetting.conv1(state)
        maxpool1 = NetworkSetting.maxpool1(self.conv1)
        self.conv2 = NetworkSetting.conv2(maxpool1)
        maxpool2 = NetworkSetting.maxpool2(self.conv2)
        reshape = NetworkSetting.reshape(maxpool2)
        rnn ,l ,_ = NetworkSetting.lstm(reshape, state)
        drop = NetworkSetting.dropout(rnn, keep_prob)
        fc1 = NetworkSetting.fc1(drop)
        
        q_value = NetworkSetting.q_value(fc1)
        
        return q_value

    def _build_graph(self):
        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        
        self.prob = tf.nn.softmax(self.q_model, axis=1)
        
        self.onehot = tf.one_hot(self.target_, depth=64)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.q_model, labels=self.onehot)
        self.loss = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer()
        self.update_step = optimizer.minimize(self.loss)
    
    def update_parameter_server_batch(self, s1, target):
        if np.ndim(s1) == 4:
            s1 = np.array([s1])            
        
        assert np.shape(s1) == (BATCH_SIZE, N_ADV)+RESOLUTION, print(np.shape(s1))
        weights = SESS.run(self.weights_params)
        assert np.isnan([np.mean(w) for w in weights]).any()==False , print(weights)
        feed_dict = {self.state1_: s1, self.target_:target, self.keep_prob_:KEEP_PROB}
        l,_ = SESS.run([self.loss, self.update_step], feed_dict)
        return l

    def predict_action(self, s1):
        if np.ndim(s1) == 4:
            s1 = np.array([s1])
            probs = SESS.run(self.prob, {self.state1_:s1, self.keep_prob_:1.0})
            return np.random.choice(64, p=probs[0])
        elif np.ndim(s1) == 5:
            probs = SESS.run(self.prob, {self.state1_:s1, self.keep_prob_:1.0})
            return [np.random.choice(64, p=p) for p in probs]
        else:
            return None
        
    def get_q_values(self, s1):
        if np.ndim(s1) == 4:
            s1 = np.array([s1])
            q = SESS.run(self.q_model, {self.state1_:s1, self.keep_prob_:1.0})
            return q[0]
        elif np.ndim(s1) == 5:
            q = SESS.run(self.q_model, {self.state1_:s1, self.keep_prob_:1.0})
            return q
        else:
            return None
        
    def get_loss(self, s1, target):
        if np.ndim(s1) == 4:
            s1 = np.array([s1])
            q = SESS.run(self.loss, {self.state1_:s1, self.keep_prob_:1.0, self.target_:target})
            return q[0]
        elif np.ndim(s1) == 5:
            q = SESS.run(self.loss, {self.state1_:s1, self.keep_prob_:1.0, self.target_:target})
            return q
        else:
            return None
    
    def save_model(self, model_path):
        return self.saver.save(SESS, model_path)
    
    def load_model(self, model_path):
        return self.saver.resore(SESS, model_path)


# In[ ]:


class NetworkSetting:
    
    def conv1(pre_layer):
        num_outputs = 32
        kernel_size = [1,6,6]
        stride = [1,3,3]
#         kernel_size = [6,6]
#         stride = [3,3]
        padding = 'SAME'
        activation = tf.nn.relu
        weights_init = tf.contrib.layers.xavier_initializer_conv2d()
#         weights_init = tf.constant_initializer(2.0)
        bias_init = tf.constant_initializer(0.1)
#         print(pre_layer.get_shape())
        return tf.contrib.layers.conv2d(pre_layer,kernel_size=kernel_size,                                        num_outputs=num_outputs,                                        stride=stride,padding=padding,activation_fn=activation,                                        weights_initializer=weights_init,                                        biases_initializer=bias_init)
    
    def maxpool1(pre_layer):
#         return tf.nn.max_pool(pre_layer,[1,3,3,1],[1,2,2,1],'SAME')
        return tf.nn.max_pool3d(pre_layer,[1,1,3,3,1],[1,1,2,2,1],'SAME')
    
    def conv2(pre_layer):
        num_outputs = 32
        kernel_size = [1,3,3]
        stride = [1,2,2]
#         kernel_size = [3,3]
#         stride = [2,2]
        padding = 'SAME'
        activation = tf.nn.relu
        weights_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(0.1)
        return tf.contrib.layers.conv2d(pre_layer,kernel_size=kernel_size,num_outputs=num_outputs,                                        stride=stride,padding=padding,activation_fn=activation,                                        weights_initializer=weights_init,biases_initializer=bias_init)
    
    def maxpool2(pre_layer):
#         return tf.nn.max_pool(pre_layer,[1,3,3,1],[1,2,2,1],'SAME')
        return tf.nn.max_pool3d(pre_layer,[1,1,3,3,1],[1,1,2,2,1],'SAME')
        
    def reshape(pre_layer):
        shape = pre_layer.get_shape()
        return tf.reshape(pre_layer, shape=(-1, shape[1],shape[2]*shape[3]*shape[4]))
#         return tf.reshape(pre_layer, shape=(-1, N_ADV * 2240))
    
    def lstm(pre_layer, state):
        batch_size = tf.shape(pre_layer)[0]
        temp = tf.reduce_max(state, axis=4)
        temp = tf.reduce_max(temp, axis=3)
        temp = tf.reduce_max(temp, axis=2)
        lengh = tf.cast(tf.reduce_sum(tf.sign(temp) , axis=1),dtype=tf.int32)
        lengh = tf.where(tf.equal(lengh, tf.zeros_like(lengh)), tf.ones_like(lengh), lengh)
        cell = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE)
        rnn_state = cell.zero_state(batch_size, dtype=tf.float32)
        rnn_out, state_out = tf.nn.dynamic_rnn(cell, pre_layer, initial_state=rnn_state, sequence_length=lengh,dtype=tf.float32)
        out_idx = tf.range(0, batch_size) * N_ADV + (lengh  -1)
        output = tf.gather(tf.reshape(rnn_out, [-1, LSTM_SIZE]), out_idx)
        return output, lengh, rnn_out
    
    def dropout(pre_layer, keep_prob):
        return tf.nn.dropout(pre_layer, keep_prob)
    
    def fc1(pre_layer):
        num_outputs = 64
        activation_fn = tf.nn.relu
        weights_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.1)
        return tf.contrib.layers.fully_connected(pre_layer,num_outputs=num_outputs,activation_fn=activation_fn,                                                 weights_initializer=weights_init, biases_initializer=bias_init)
    
    def q_value(pre_layer):
        num_outputs =64
        activation_fn = tf.nn.relu
        weights_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.1)
        return tf.contrib.layers.fully_connected(pre_layer,num_outputs=num_outputs,activation_fn=activation_fn,                                                 weights_initializer=weights_init, biases_initializer=bias_init)


# In[ ]:


class LogRecorder(object):
    def __init__(self,log_dir):
        
        self.state1_ = tf.placeholder(tf.float32,shape=(None,N_ADV,)+RESOLUTION, name="state1")
        with tf.variable_scope("log_recorder", reuse=tf.AUTO_REUSE):
            self.conv1, self.conv2, self.model = self._build_model(self.state1_)
        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="log_recorder")
        
        # Summary for LOSS
        self.loss_ = [tf.placeholder(tf.float32,shape=())]
        self.loss_name = ["loss_cross_entropy"]        
        self.merged_loss = self._build_scalar_summary(self.loss_, self.loss_name, "loss")
        
        # Summary for  SCORE
        self.accuracy_ = [tf.placeholder(tf.float32, shape=())]
        self.accuracy_name = ["accuracy"]
        self.merged_accuracy = self._build_scalar_summary(self.accuracy_, self.accuracy_name, "score")
 
        # Summary for SCREEN
        self.display_state1 = tf.squeeze(tf.gather(self.state1_, indices=[0], axis=0))
        self.display_conv1 = tf.gather(tf.squeeze(tf.gather(self.conv1, indices=[0], axis=0)), indices=[0], axis=3)
        self.display_conv2 = tf.gather(tf.squeeze(tf.gather(self.conv2, indices=[0], axis=0)), indices=[0], axis=3)
        image_shapes = [self.display_state1.get_shape(), self.display_conv1.get_shape(), self.display_conv2.get_shape()]
        image_name = ["state1", "conv1", "conv2"]        
        self.merged_images = self._build_image_summary([self.display_state1, self.display_conv1, self.display_conv2], N_ADV, image_name, "states")
        
        #Sumamry for FILTER
        display_filter_conv1 = tf.reshape(tf.transpose(self.weights_params[0], [3,4,0,1,2]), (-1,self.weights_params[0].get_shape()[1], self.weights_params[0].get_shape()[2]))
        display_filter_conv2 = tf.reshape(tf.transpose(self.weights_params[2], [3,4,0,1,2]), (-1,self.weights_params[2].get_shape()[1], self.weights_params[2].get_shape()[2]))
        display_filter_conv1 = tf.expand_dims(display_filter_conv1, -1)
        display_filter_conv2 = tf.expand_dims(display_filter_conv2, -1)
        filter_shapes = [display_filter_conv1.get_shape(), display_filter_conv2.get_shape()]
        filter_names = ["conv1", "conv2"]
        self.merged_filters = self._build_image_summary([display_filter_conv1, display_filter_conv2], 10, filter_names, "filters")
        
        self.writer = tf.summary.FileWriter(log_dir,SESS.graph)
    
    def _build_model(self,state):
        conv1 = NetworkSetting.conv1(state)
        maxpool1 = NetworkSetting.maxpool1(conv1)
        conv2 = NetworkSetting.conv2(maxpool1)
        maxpool2 = NetworkSetting.maxpool2(conv2)
        reshape = NetworkSetting.reshape(maxpool2)
        rnn ,l ,_ = NetworkSetting.lstm(reshape, state)
        fc1 = NetworkSetting.fc1(rnn)
        
        q_value = NetworkSetting.q_value(fc1)
        
        return conv1, conv2,q_value
    
    def _build_scalar_summary(self, placeholders, names, family):
        return tf.summary.merge([tf.summary.scalar(n, i, family=family) for n,i in zip(names, placeholders)])
    
    def _build_image_summary(self, placeholders,n_output, names, family):
        summaries = []
        for i, p in enumerate(placeholders):
            shape = p.get_shape().as_list()
            summ = tf.summary.image(names[i], tf.reshape(p, [-1, shape[1], shape[2], shape[3]]), n_output, family=family)
            summaries.append(summ)
        return tf.summary.merge(summaries) 
        
    def write_loss(self, step, loss):
        m = SESS.run(self.merged_loss, {self.loss_[0]:loss})
        return self.writer.add_summary(m, step)
    
    def write_accuracy(self, step, acc):
        m = SESS.run(self.merged_accuracy, {self.accuracy_[0]:acc})
        return self.writer.add_summary(m, step)
    
    def write_images(self, step, s1):
        feed_dict = {self.state1_:s1}
        m = SESS.run(self.merged_images, feed_dict)
        return self.writer.add_summary(m, step)
    
    def write_filters(self, step):
        m = SESS.run(self.merged_filters)
        return self.writer.add_summary(m, step)
    
    def copy_weights(self, network):
        SESS.run([i.assign(j) for i,j in zip(self.weights_params, network.weights_params)])


# In[ ]:


if __name__ == "learning":
    
    TEST_LOSS = []
    TRAIN_LOSS = []
    
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list="0"))
    config.log_device_placement = False
    config.allow_soft_placement = True
    SESS = tf.Session(config=config)

    with tf.device("/gpu:0"):
        network = NetworkLocal("test")
    SESS.run(tf.global_variables_initializer())

    log_rec = LogRecorder(log_dir="./logs/log_test/")
    
    img, label_row = load_data()
    label = np.array([l[-1] for l in label_row])
    label = label.astype(np.int32)
    n_data = img.shape[0]

    train_img, test_img , train_label, test_label = train_test_split(img, label, train_size=0.8, random_state=1)
    n_train = np.shape(train_img)[0]
    n_test = np.shape(test_label)[0]
    
    print(n_data)
    
    for i in tqdm(range(1000)):
        batch_idx = np.random.randint(n_train, size=BATCH_SIZE)
        batch_img = train_img[batch_idx]
        batch_label = train_label[batch_idx]
        l = network.update_parameter_server_batch(batch_img, batch_label)
        if (i+1) % 10 == 0:
            TRAIN_LOSS.append(l)
            log_rec.copy_weights(network)
            batch_idx = np.random.randint(n_test, size=50)
            predicted = network.predict_action(test_img[batch_idx])
            acc = sum(predicted == test_label[batch_idx]) / 50
            log_rec.write_loss(i, loss=l)
            log_rec.write_accuracy(i,acc=acc)
            log_rec.write_images(i, [img[200]])
            log_rec.write_filters(i)
            l = network.get_loss(test_img[batch_idx], test_label[batch_idx])
            TEST_LOSS.append(l)


# In[ ]:


network.save_model("./models/model_predict_action/model.ckpt")


# In[ ]:


idx = np.random.randint(n_train, size=100)
predict = network.predict_action(train_img[idx])
acc_train = sum(predict==train_label[idx]) / 100


# In[ ]:


idx = np.random.randint(n_test, size=100)
predict = network.predict_action(test_img[idx])
acc_test = sum(predict==test_label[idx]) / 100


# In[ ]:


acc_train


# In[ ]:


acc_test


# In[ ]:


plt.plot(range(len(TEST_LOSS)), TEST_LOSS, "b")
plt.plot(range(len(TEST_LOSS)), TRAIN_LOSS, "r")


# In[ ]:


TEST_LOSS

