#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from vizdoom import *
import os, time, random, threading, h5py, math,pickle, sys
import tensorflow as tf
import numpy as np


# In[ ]:


class Agent(object):
    
    def __init__(self, network,random_seed, parameters):
        self.network = network
        self.randomstate = np.random.RandomState(random_seed)
        self.parameters = parameters
        
    def calc_eps(self, progress):
        if progress < 0.2:
            return self.parameters.eps_min
        elif progress >= 0.2 and progress < 0.8:
            return (( self.parameters.eps_max - self.parameters.eps_min)/ 0.6) * progress + ( self.parameters.eps_min -  (self.parameters.eps_max - self.parameters.eps_min)/ 0.6 * 0.2)
        else :
            return self.parameters.eps_max

    def act_eps_greedy(self, sess, s1, progress):
        assert progress >= 0.0 and progress <=1.0
        
        eps = self.calc_eps(progress)
        if self.randomstate.rand() <= eps:
            a_idx = self.randomstate.choice(range(self.parameters.n_agent_action), p=self.network.get_policy(sess,[s1])[0])
#             a_idx = self.network.get_best_action(sess, [s1])[0]
        else:
            a_idx = self.randomstate.randint(self.parameters.n_agent_action)
            
        return a_idx
    
    def act_greedy(self, sess, s1):
        a_idx = self.randomstate.choice(range(self.parameters.n_agent_action), p=self.network.get_policy(sess,[s1])[0])
#         a_idx = self.network.get_best_action(sess, [s1])[0]
        return a_idx
    
    def get_sum_prob(self,sess, s1):
        q_value = self.network.get_qvalue_learning(sess, [s1])[0]
        q_value = np.maximum(q_value,0) + 0.01
        q_prob = (q_value)/sum(q_value)
        a_idx = np.random.choice(self.parameters.n_agent_action, p=q_prob)
        return a_idx

