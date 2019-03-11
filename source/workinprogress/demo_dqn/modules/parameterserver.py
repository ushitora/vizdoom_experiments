#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from vizdoom import *
import os, time, random, threading, h5py, math,pickle, sys
import tensorflow as tf
import numpy as np


# In[ ]:


class ParameterServer:
    def __init__(self, sess, log_dir, networksetting, parameters):
        self.sess = sess
        self.parameters = parameters
        with tf.variable_scope("parameter_server", reuse=tf.AUTO_REUSE):
            self.state1_ = tf.placeholder(tf.float32, shape=(None,) + self.parameters.resolution)
            self.q_value, self.conv1, self.conv2, self.q_prob = self._build_model(self.state1_, networksetting)

        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="parameter_server")
#         self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, RMSProbDecaly)
        self.optimizer = tf.train.AdamOptimizer()
            
        with tf.variable_scope("summary", reuse=tf.AUTO_REUSE):
            self._build_summary(sess,log_dir)
        
        self.saver = tf.train.Saver(max_to_keep = 20)
    

    def _build_model(self,state,networksetting):
            conv1 = networksetting.conv1(state)
            maxpool1 = networksetting.maxpool1(conv1)
            conv2 = networksetting.conv2(maxpool1)
            maxpool2 = networksetting.maxpool2(conv2)
            reshape = networksetting.reshape(maxpool2)
            fc1 = networksetting.fc1(reshape)
            q = networksetting.q_value(fc1)
            
            q_prob = tf.nn.softmax(q)
                
            print("---------MODEL SHAPE-------------")
            print(state.get_shape())
            print(conv1.get_shape())
            print(maxpool1.get_shape())
            print(conv2.get_shape())
            print(maxpool2.get_shape())
            print(reshape.get_shape())
            print(fc1.get_shape())
            print(q.get_shape())
            
            return q, conv1, conv2, q_prob
                
    def _build_summary(self,sess, log_dir):
        
        self.reward_ = tf.placeholder(tf.float32,shape=(), name="reward")
        self.frag_ = tf.placeholder(tf.float32, shape=(), name="frag")
        self.death_ = tf.placeholder(tf.float32, shape=(), name="death")
        self.kill_ = tf.placeholder(tf.float32, shape=(), name="kill")
        self.score_step_ = tf.placeholder(tf.float32, shape=(), name="step")
        self.score_processtime_ = tf.placeholder(tf.float32, shape=(), name="score_processtime")
        self.loss_one_ = tf.placeholder(tf.float32, shape=(), name="loss_one")
        self.loss_adv_ = tf.placeholder(tf.float32, shape=(), name="loss_adv")
        self.loss_cls_ = tf.placeholder(tf.float32, shape=(), name="loss_class")
        self.loss_l2_ = tf.placeholder(tf.float32, shape=(), name="loss_l2")
        self.loss_processtime_ = tf.placeholder(tf.float32, shape=(), name="loss_processtime")
        
        with tf.variable_scope("Summary_Score"):
            s = [tf.summary.scalar('reward', self.reward_, family="score"), tf.summary.scalar('frag', self.frag_, family="score"),                  tf.summary.scalar("death", self.death_, family="score"),
                 tf.summary.scalar("kill", self.kill_, family="score"), \
                 tf.summary.scalar("step",self.score_step_, family="score"), 
                 tf.summary.scalar("score_processtime", self.score_processtime_, family="score")]
            self.summary_reward = tf.summary.merge(s)
        
        with tf.variable_scope("Summary_Loss"):
            list_summary = [tf.summary.scalar('loss_onestep', self.loss_one_, family="loss"), 
                            tf.summary.scalar('loss_advantage', self.loss_adv_, family="loss"),
                            tf.summary.scalar('loss_class', self.loss_cls_, family="loss"),
                            tf.summary.scalar('loss_l2', self.loss_l2_, family='loss'),
                            tf.summary.scalar('loss_processtime', self.loss_processtime_, family="loss")]
            self.summary_loss = tf.summary.merge(list_summary)
        
#         with tf.variable_scope("Summary_Images"):
#             conv1_display = tf.reshape(tf.transpose(self.conv1, [0,1,4,2,3]), (-1, self.conv1.get_shape()[1],self.conv1.get_shape()[2]))
#             conv2_display = tf.reshape(tf.transpose(self.conv2, [0,1,4,2,3]), (-1, self.conv2.get_shape()[1],self.conv2.get_shape()[2]))
#             conv1_display = tf.expand_dims(conv1_display, -1)
#             conv2_display = tf.expand_dims(conv2_display, -1)

#             state_shape = self.state1_.get_shape()
#             conv1_shape = conv1_display.get_shape()
#             conv2_shape = conv2_display.get_shape()

#             s_img = []
#             s_img.append(tf.summary.image('state',tf.reshape(self.state1_,[-1, state_shape[2], state_shape[3], state_shape[4]]), 1, family="state1"))
#             s_img.append(tf.summary.image('conv1',tf.reshape(self.conv1,[-1, conv1_shape[1], conv1_shape[2], 1]), family="conv1"))
#             s_img.append(tf.summary.image('conv2',tf.reshape(self.conv2,[-1, conv2_shape[1], conv2_shape[2], 1]), family="conv2"))

#             self.summary_image = tf.summary.merge(s_img)
            
        with tf.variable_scope("Summary_Weights"):
            s = [tf.summary.histogram(values=w, name=w.name, family="weights") for w in self.weights_params]
            self.summary_weights = tf.summary.merge(s)

        self.writer = tf.summary.FileWriter(log_dir)
        
    def write_graph(self, sess):
        self.writer.add_graph(sess.graph)
        
    def write_score(self,sess, step ,reward, frag, death, kill, score_step, processtime):
        m = sess.run(self.summary_reward, feed_dict={self.reward_:reward, self.frag_:frag, self.death_:death, self.kill_:kill, self.score_step_:score_step, self.score_processtime_:processtime})
        return self.writer.add_summary(m, step)
    
    def write_loss(self,sess, step, l_o, l_n,l_c, l_l, process_time):
        m = sess.run(self.summary_loss, feed_dict={self.loss_one_: l_o, self.loss_adv_:l_n, self.loss_cls_:l_c, self.loss_l2_:l_l, self.loss_processtime_:processtime})
        return self.writer.add_summary(m, step)
    
#     def write_img(self,sess, step, state):
#         m = sess.run(self.summary_image, feed_dict={self.state1_: state})
#         return self.writer.add_summary(m, step)
    
    def write_weights(self, sess, step):
        m = sess.run(self.summary_weights)
        return self.writer.add_summary(m, step)
        
    def load_model(self, sess, model_path, step):
        self.saver.restore(sess, model_path+'-'+str(step))
    
    def save_model(self, sess,  model_path, step):
        self.saver.save(sess, model_path, global_step = step)
        
    def load_cnnweights(self, sess, weights_path):
        assert len(weights_path) == 4
        cnn_weights = self.weights_params[:4]
        w_demo = [np.load(w_p) for w_p in weights_path]
        plh = [tf.placeholder(tf.float32, shape=w.shape) for w in w_demo]
        assign_op = [w.assign(p) for w, p in zip(cnn_weights, plh)]
        feed_dict = {p:w for w,p in zip(w_demo, plh)}
        sess.run(assign_op, feed_dict)

