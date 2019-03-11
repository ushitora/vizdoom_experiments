#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from vizdoom import *
import os, time, random, threading, h5py, math,pickle, sys
import tensorflow as tf
import numpy as np


# In[ ]:


class NetworkLocal(object):
    def __init__(self,name, parameter_server, networksetting, parameters):
        self.name = name
        self.parameters = parameters
        
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("learning_network"):
                self.state1_ = tf.placeholder(tf.float32,shape=(None,)+self.parameters.resolution, name="state_1")
                self.q_value, self.conv1, self.conv2,self.reshape,self.fc1 = self._build_model(self.state1_,networksetting)
            with tf.variable_scope("target_network"):
                self.state1_target_ = tf.placeholder(tf.float32,shape=(None,)+self.parameters.resolution, name="state_1")
                self.q_value_target,_,_,_,_ = self._build_model(self.state1_target_,networksetting)
            
            self.a_ = tf.placeholder(tf.int32, shape=(None,), name="action")
            self.target_one_ = tf.placeholder(tf.float32, shape=(None,), name="target_one_")
            self.target_n_ = tf.placeholder(tf.float32, shape=(None,), name="target_n_")
            self.isdemo_ = tf.placeholder(tf.float32,shape=(None,), name="isdemo_")
            self.mergin_ = tf.placeholder(tf.float32,shape=(None,self.parameters.n_agent_action), name="mergin_")
            self.is_weight_ = tf.placeholder(tf.float32, shape=(None,), name="is_weight")
            
            self._build_graph()
        
        self.update_global_weight_params =             parameter_server.optimizer.apply_gradients([(g,w) for g, w in zip(self.grads, parameter_server.weights_params)])
        
        self.pull_global_weight_params = [l_p.assign(g_p) for l_p,g_p in zip(self.weights_params_learning,parameter_server.weights_params)]
        self.push_local_weight_params = [g_p.assign(l_p) for g_p,l_p in zip(parameter_server.weights_params,self.weights_params_learning)]

    def _build_model(self,state,networksetting):
        conv1 = networksetting.conv1(state)
        maxpool1 = networksetting.maxpool1(conv1)
        conv2 = networksetting.conv2(maxpool1)
        maxpool2 = networksetting.maxpool2(conv2)
        reshape = networksetting.reshape(maxpool2)
        fc1 = networksetting.fc1(reshape)
        
        q_value = networksetting.q_value(fc1)
        
        return q_value, conv1, conv2,reshape,fc1

    def _build_graph(self):

        self.q_prob = tf.nn.softmax(self.q_value)
        self.q_argmax = tf.argmax(self.q_value, axis=1)
        self.q_learning_max = tf.reduce_max(self.q_value, axis=1)
        self.q_target_max = tf.reduce_max(self.q_value_target, axis=1)
        
        action_idxlist = tf.transpose([tf.range(tf.shape(self.q_value)[0]), self.a_])
        
        self.weights_params_learning = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+"/learning_network")
        self.weights_params_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+"/target_network")
        
        self.tderror_one = self.parameters.lambda1 * tf.abs(self.target_one_ - tf.gather_nd(self.q_value, indices=action_idxlist))
        self.loss_one = (self.parameters.lambda1 * tf.square(self.target_one_ - tf.gather_nd(self.q_value, indices=action_idxlist))) * self.is_weight_
        self.tderror_n = self.parameters.lambda2 * tf.abs(self.target_n_ - tf.gather_nd(self.q_value, indices=action_idxlist))
        self.loss_n = (self.parameters.lambda2 * tf.square(self.target_n_ - tf.gather_nd(self.q_value, indices=action_idxlist)))*self.is_weight_
        
        self.loss_l2 = self.parameters.lambda4 * tf.reduce_sum([tf.nn.l2_loss(w) for w in self.weights_params_learning])
        
        self.loss_mergin = self.parameters.lambda3 * ((tf.stop_gradient(tf.reduce_max(self.q_value + self.mergin_, axis=1)) - tf.gather_nd(self.q_value,indices=action_idxlist))*self.isdemo_)
        
        self.tderror_total = self.tderror_one + self.tderror_n
        self.loss_total = tf.reduce_mean(self.loss_one +  self.loss_n + self.loss_mergin + self.loss_l2)    
        
        self.grads = tf.gradients(self.loss_total ,self.weights_params_learning)
        
        self.copy_params = [t.assign(l) for l,t in zip(self.weights_params_learning, self.weights_params_target)]
        
    def copy_network_learning2target(self, sess):
        return sess.run(self.copy_params)
        
    def pull_parameter_server(self, sess):
        return sess.run(self.pull_global_weight_params)
    
    def push_parameter_server(self,sess):
        return sess.run(self.push_local_weight_params)
        
    def get_weights_learngin(self, sess):
        return sess.run(self.weights_params_learning)
    
    def get_weights_target(self, sess):
        return sess.run(self.weights_params_target)
    
    def get_loss(self, sess,s1, a, target_one,target_n, isdemo, is_weight):
        mergin_value = np.ones((len(s1), self.parameters.n_agent_action)) * self.parameters.mergin_value
        mergin_value[range(len(s1)), a] = 0.0
        feed_dict = {self.state1_: s1,self.a_:a, self.target_one_:target_one, self.target_n_:target_n, self.isdemo_:isdemo, self.is_weight_:is_weight, self.mergin_:mergin_value}
#         l_one, l_n, l_mergin, l_l2, tderror_total = sess.run([self.loss_one, self.loss_n, self.loss_mergin, self.loss_l2, self.tderror_total], feed_dict)
        l_one, tderror_total = sess.run([self.loss_n, self.tderror_n], feed_dict)
        return l_one, 0,0,0, tderror_total
    
    def get_losstotal(self, sess,s1, a, target_one,target_n, isdemo, is_weight):
        mergin_value = np.ones((len(s1), self.parameters.n_agent_action)) * self.parameters.mergin_value
        mergin_value[range(len(s1)), a] = 0.0
        feed_dict = {self.state1_: s1,self.a_:a, self.target_one_:target_one, self.target_n_:target_n, self.isdemo_:isdemo, self.is_weight_:is_weight, self.mergin_:mergin_value}
        loss_total = sess.run([self.loss_total], feed_dict)
        return loss_total[0]
    
    def get_grads(self, sess,s1, a, target_one,target_n, isdemo, is_weight):
        mergin_value = np.ones((len(s1), self.parameters.n_agent_action)) * self.parameters.mergin_value
        mergin_value[range(len(s1)), a] = 0.0
        feed_dict = {self.state1_: s1,self.a_:a, self.target_one_:target_one, self.target_n_:target_n, self.isdemo_:isdemo, self.is_weight_:is_weight, self.mergin_:mergin_value}
        grads = sess.run(self.grads, feed_dict)
        return grads
    
    def update_parameter_server(self, sess, s1, a, target_one,target_n, isdemo, is_weight):
        assert np.ndim(s1) == 4
        mergin_value = np.ones((len(s1), self.parameters.n_agent_action)) * self.parameters.mergin_value
        mergin_value[range(len(s1)), a] = 0.0
        feed_dict = {self.state1_: s1,self.a_:a, self.target_one_:target_one, self.target_n_:target_n, self.isdemo_:isdemo, self.is_weight_:is_weight, self.mergin_:mergin_value}
#         _,l_one, l_n, l_mergin, l_l2, tderror_total = sess.run([self.update, self.loss_one, self.loss_n, self.loss_mergin, self.loss_l2, self.tderror_total], feed_dict)
#         _,l_one,l_mergin,l_l2 ,tderror_total = sess.run([self.update_global_weight_params,self.loss_n,self.loss_mergin,self.loss_l2, self.tderror_total], feed_dict)
#         return l_one, 0,l_mergin,l_l2, tderror_total
        _,l_one, l_n, l_mergin, l_l2, tderror_total = sess.run([self.update_global_weight_params, self.loss_one, self.loss_n, self.loss_mergin, self.loss_l2, self.tderror_total], feed_dict)
        return l_one, l_n, l_mergin, l_l2, tderror_total
    
    def check_weights(self, sess):
        weights = sess.run(self.weights_params_learning)
        assert np.isnan([np.mean(w) for w in weights]).any()==False , print(weights)

    def get_qvalue_learning(self, sess, s1):
        assert np.ndim(s1) == 4
        return sess.run(self.q_value, {self.state1_: s1})
    
    def get_qvalue_lerning_max(self, sess, s1):
        return sess.run(self.q_learing_max, {self.state1_:s1})

    def get_qvalue_target(self, sess ,s1):
        assert np.ndim(s1) == 4
        return sess.run(self.q_value_target, {self.state1_target_:s1})
    
    def get_qvalue_target_max(self, sess, s1):
        assert np.ndim(s1) == 4
        return sess.run(self.q_target_max, {self.state1_target_:s1})
    
    def get_qvalue_max_learningaction(self, sess, s1):
        assert np.ndim(s1) == 4
        action_idx, q_value = sess.run([self.q_argmax, self.q_value_target], {self.state1_:s1, self.state1_target_:s1})
        return q_value[range(np.shape(s1)[0]), action_idx]
    
    def get_policy(self, sess, s1):
        return sess.run(self.q_prob, {self.state1_: s1})
    
    def get_best_action(self,sess, s1):
        return sess.run(self.q_argmax, {self.state1_:s1})

