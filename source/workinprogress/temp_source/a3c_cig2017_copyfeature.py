#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/python

from __future__ import print_function
import math, os
import time,random,threading
import tensorflow as tf
from time import sleep
import numpy as np
from vizdoom import *
import skimage.color, skimage.transform
from tqdm import tqdm
from tensorflow.python import debug as tf_debug

CONFIG_FILE_PATH = "./config/custom_config.cfg"
MODEL_PATH = "./model_v00/model_v00.ckpt"
RESOLUTION = (120,120,3)

N_ADV = 5

N_WORKERS = 3

WORKER_STEPS = 10000

N_ACTION = 6

GAMMA = 0.99

BOTS_NUM = 5

EPS_START = 0.5
EPS_END = 0.0
EPS_STEPS = WORKER_STEPS*N_WORKERS

REWARDS = {'living':-0.01, 'health_loss':-1, 'medkit':50, 'ammo':0.0, 'frag':500, 'dist':3e-2, 'suicide':-500} 

LEARNING_RATE = 5e-3
RMSProbDecaly = 0.99

MERGED_WEIGHTS_PATH=["./weights_merged/conv1_kernel.npy", "./weights_merged/conv1_bias.npy", "./weights_merged/conv2_kernel.npy", "./weights_merged/conv2_bias.npy"]


# In[ ]:


# --スレッドになるクラスです　-------
class Worker_thread:
    # スレッドは学習環境environmentを持ちます
    def __init__(self, thread_name, parameter_server):
        self.environment = Environment(thread_name, parameter_server,False)
        print(thread_name," Initialized")

    def run(self):
        while True:
            if not self.environment.finished:
                self.environment.run()
            else:
                sleep(1.0)


# In[ ]:


class Environment(object):
    def __init__(self,name, parameter_server,record=False):
        self.game = DoomGame()
        self.game.load_config(CONFIG_FILE_PATH)
        self.game.set_window_visible(False)
        self.game.set_mode(Mode.PLAYER)
#         self.game.set_screen_format(ScreenFormat.GRAY8)
        self.game.set_screen_format(ScreenFormat.CRCGCB)
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.init()
        
        health = self.game.get_game_variable(GameVariable.HEALTH)
        ammo = self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        frag = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        pos_x = self.game.get_game_variable(GameVariable.POSITION_X)
        pos_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        self.reward_gen = RewardGenerater(health,ammo,frag,pos_x,pos_y)
        
        self.network = Network_local(name, parameter_server)
        self.agent = Agent(name,self.network)
        
        self.local_step = 0
        
        self.finished = False
        
        self.name = name
        
        self.record = record
        
        self.parameter_server = parameter_server
        
        self.frame_record = []
        self.loss_policy = []
        self.loss_value = []
        self.entropy = []
    
    def start_episode(self):
        self.game.new_episode()
        for i in range(BOTS_NUM):
            self.game.send_game_command("addbot")
        
    def preprocess(self,img):
        if len(img.shape) == 3:
            img = img.transpose(1,2,0)

        img = skimage.transform.resize(img, RESOLUTION,mode='constant')
        img = img.astype(np.float32)
        return img
    
    def get_reward(self):
        health = self.game.get_game_variable(GameVariable.HEALTH)
        ammo = self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        frag = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        pos_x = self.game.get_game_variable(GameVariable.POSITION_X)
        pos_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        
        r,r_detail = self.reward_gen.get_reward(health,ammo,frag,pos_x,pos_y)
    
        return r
    
    def run(self):
        global frames
        
        self.start_episode()
        
        train_episode = 0
        record_l_p = 0
        record_l_v = 0
        for step in range(WORKER_STEPS):
            
#             if step % 1000 == 0:
#                 buff = self.name + ":" + str(step) + "step is passed"
#                 print(buff)
            #Copy params from global
            self.agent.network.pull_parameter_server()

            if not self.game.is_episode_finished():
                
                if step%N_ADV==0 and not step==0:
                    self.reward_gen.update_origin(self.game.get_game_variable(GameVariable.POSITION_X),                                                  self.game.get_game_variable(GameVariable.POSITION_Y))

                s1 = self.preprocess(self.game.get_state().screen_buffer)
                action = self.agent.act(s1)
                self.game.make_action(action,5)
                reward = self.get_reward()
                isterminal = self.game.is_episode_finished()
                s2 = self.preprocess(self.game.get_state().screen_buffer) if not isterminal else None
                
                if self.record==True:
                    if l_p != 0:
                        record_l_p = l_p
                    if l_v != 0:
                        record_l_v = l_v
                        
                    self.parameter_server.write_loss(frames, record_l_p, recordl_v)
                    if step %100 == 0:
                        self.parameter_server.write_image(s1)

                l_p, l_v = self.agent.advantage_push_network(s1,action,reward,s2,isterminal)
                
                if self.game.is_player_dead():
                    self.game.respawn_player()
                    self.reward_gen.respawn_pos(self.game.get_game_variable(GameVariable.HEALTH),                                                 self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO),                                                 self.game.get_game_variable(GameVariable.POSITION_X),                                                self.game.get_game_variable(GameVariable.POSITION_Y))

            else:
                train_episode += 1
                self.start_episode()
                self.reward_gen.new_episode(health = self.game.get_game_variable(GameVariable.HEALTH),                                            ammo = self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO),                                            posx = self.game.get_game_variable(GameVariable.POSITION_X),                                            posy = self.game.get_game_variable(GameVariable.POSITION_Y))
            self.local_step += 1   
            frames += 1
                
        print(self.name," finished")
        self.finished = True


# In[ ]:


class TestEnvironment(object):
    def __init__(self,name, parameter_server):
        self.game = DoomGame()
        self.game.load_config(CONFIG_FILE_PATH)
        self.game.set_window_visible(False)
        self.game.set_mode(Mode.PLAYER)
#         self.game.set_screen_format(ScreenFormat.GRAY8)
        self.game.set_screen_format(ScreenFormat.CRCGCB)
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.init()
        
        health = self.game.get_game_variable(GameVariable.HEALTH)
        ammo = self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        frag = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        pos_x = self.game.get_game_variable(GameVariable.POSITION_X)
        pos_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        self.reward_gen = RewardGenerater(health,ammo,frag,pos_x,pos_y)
        
        self.network = Network_local(name, parameter_server)
        self.agent = Agent(name,self.network)
        
        self.pre_death = 0
        
        self.record_reward = []
        self.record_frag = []
        self.record_death = []
    
    def start_episode(self):
        self.game.new_episode()
        for i in range(BOTS_NUM):
            self.game.send_game_command("addbot")
        
    def preprocess(self,img):
        if len(img.shape) == 3:
            img = img.transpose(1,2,0)

        img = skimage.transform.resize(img, RESOLUTION,mode='constant')
        img = img.astype(np.float32)
        return img
    
    def get_reward(self):
        health = self.game.get_game_variable(GameVariable.HEALTH)
        ammo = self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        frag = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        pos_x = self.game.get_game_variable(GameVariable.POSITION_X)
        pos_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        
        r,r_detail = self.reward_gen.get_reward(health,ammo,frag,pos_x,pos_y)
    
        return r
    
    def run(self):

        global frames
        health = self.game.get_game_variable(GameVariable.HEALTH)
        ammo = self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        frag = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        pos_x = self.game.get_game_variable(GameVariable.POSITION_X)
        pos_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        self.reward_gen = RewardGenerater(health,ammo,0,pos_x,pos_y)
        
        self.start_episode()
        
        #Copy params from global
        self.agent.network.pull_parameter_server()

        step = 0
        while not self.game.is_episode_finished():
            
            if step%N_ADV==0 and not step==0:
                self.reward_gen.update_origin(self.game.get_game_variable(GameVariable.POSITION_X),                                              self.game.get_game_variable(GameVariable.POSITION_Y))

            s1 = self.preprocess(self.game.get_state().screen_buffer)
            action = self.agent.act(s1)
            self.game.make_action(action,1)
            reward = self.get_reward()
            isterminal = self.game.is_episode_finished()

            if self.game.is_player_dead():
                self.game.respawn_player()
                self.reward_gen.respawn_pos(self.game.get_game_variable(GameVariable.HEALTH),                                             self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO),                                             self.game.get_game_variable(GameVariable.POSITION_X),                                            self.game.get_game_variable(GameVariable.POSITION_Y))
            
            step += 1
                
        print("----------TEST at %d step-------------"%(frames))
        print("FRAG:",self.game.get_game_variable(GameVariable.FRAGCOUNT),"DEATH:",self.game.get_game_variable(GameVariable.DEATHCOUNT)-self.pre_death)
        print("REWARD",self.reward_gen.total_reward)
        print("DETAIL:",self.reward_gen.total_reward_detail)
        self.record_frag.append(self.game.get_game_variable(GameVariable.FRAGCOUNT))
        self.record_death.append(self.game.get_game_variable(GameVariable.DEATHCOUNT)-self.pre_death)
        self.record_reward.append(self.reward_gen.total_reward)
        self.pre_death = self.game.get_game_variable(GameVariable.DEATHCOUNT)


# In[ ]:


class RewardGenerater(object):
    def __init__(self,health,ammo,frag,pos_x,pos_y):

        # Reward
        self.rewards = REWARDS
        self.dist_unit = 6.0
        
        self.origin_x = pos_x
        self.origin_y = pos_y
        
        self.pre_health = health
        self.pre_ammo = ammo
        self.pre_frag = frag

        self.total_reward = 0.0
        self.total_reward_detail = {'living':0.0, 'health_loss':0.0, 'medkit':0.0, 'ammo':0.0, 'frag':0.0, 'dist':0.0, 'suicide': 0.0}

    
    def get_reward(self,health,ammo,frag,pos_x,pos_y):
        
        if abs(health) > 10000:
            health = 100.0

        if self.origin_x == 0 and self.origin_y == 0:
            self.origin_x = pos_x
            self.origin_y = pos_y
        
        self.reward_detail = self.calc_reward(frag-self.pre_frag,0.0,                                               health-self.pre_health,                                              ammo-self.pre_ammo,                                               pos_x-self.origin_x,                                               pos_y-self.origin_y)
        self.reward = sum(self.reward_detail.values())

        for k,v in self.reward_detail.items():
            self.total_reward_detail[k] += v
        self.total_reward = sum(self.total_reward_detail.values())

        self.pre_frag = frag
        self.pre_health = health
        self.pre_ammo = ammo
                    
        return (self.reward, self.reward_detail)
    
    def calc_reward(self,m_frag,m_death,m_health,m_ammo,m_posx,m_posy):

        ret_detail = {}

        ret_detail['living'] = self.rewards['living']

        if m_frag >= 0:
            ret_detail['frag'] = (m_frag)*self.rewards['frag']
            ret_detail['suicide'] = 0.0
        else:
            ret_detail['suicide'] = (m_frag*-1)*(self.rewards['suicide'])
            ret_detail['frag'] = 0.0
        
        ret_detail['dist'] = int((math.sqrt((m_posx)**2 + (m_posy)**2))/self.dist_unit) * (self.rewards['dist'] * self.dist_unit)
        
        if m_health > 0:
            ret_detail['medkit'] = self.rewards['medkit']
            ret_detail['health_loss'] = 0.0
        else:
            ret_detail['medkit'] = 0.0
            ret_detail['health_loss'] = (m_health)*self.rewards['health_loss'] * (-1)

        ret_detail['ammo'] = (m_ammo)*self.rewards['ammo'] if m_ammo>0 else 0.0
        
        return ret_detail 
    
    def respawn_pos(self,health,ammo,posx, posy):
        self.origin_x = posx
        self.origin_y = posy
        self.pre_health = health
        self.pre_ammo = ammo

    def new_episode(self,health,ammo,posx,posy):
        self.respawn_pos(health,ammo,posx,posy)
        self.pre_frag = 0

        self.total_reward = 0
        self.total_reward_detail={'living':0.0, 'health_loss':0.0, 'medkit':0.0, 'ammo':0.0, 'frag':0.0, 'dist':0.0, 'suicide': 0.0}
    
    def update_origin(self,pos_x, pos_y):
        self.origin_x = pos_x
        self.origin_y = pos_y


# In[ ]:


class NetworkSetting:
    
    def state():
        name = "STATE"
        shape = [None,RESOLUTION[0],RESOLUTION[1],RESOLUTION[2]]
        return tf.placeholder(tf.float32,shape=shape,name=name)
    
    def conv1(pre_layer):
        num_outputs = 32
        kernel_size = [6,6]
        stride = 3
        padding = 'SAME'
        activation = tf.nn.relu
        weights_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(0.1)
        
        return tf.contrib.layers.conv2d(pre_layer,kernel_size=kernel_size,num_outputs=num_outputs,                                             stride=stride,padding=padding,activation_fn=activation,                                            weights_initializer=weights_init,                                             biases_initializer=bias_init)
    def maxpool1(pre_layer):
        return tf.nn.max_pool(pre_layer,[1,3,3,1],[1,2,2,1],'SAME')
    
    def conv2(pre_layer):
        num_outputs = 32
        kernel_size = [3,3]
        stride = 2
        padding = 'SAME'
        activation = tf.nn.relu
        weights_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(0.1)
        return tf.contrib.layers.conv2d(pre_layer,kernel_size=kernel_size,num_outputs=num_outputs,                                             stride=stride,padding=padding,activation_fn=activation,                                            weights_initializer=weights_init,biases_initializer=bias_init)
    
    def maxpool2(pre_layer):
        return tf.nn.max_pool(pre_layer,[1,3,3,1],[1,2,2,1],'SAME')
        
    def reshape(pre_layer):
        return tf.contrib.layers.flatten(pre_layer)
        
    def fc1(pre_layer):
        num_outputs = 512
        activation_fn = tf.nn.relu
        weights_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.1)
        return tf.contrib.layers.fully_connected(pre_layer,num_outputs=num_outputs,activation_fn=activation_fn,                                                    weights_initializer=weights_init, biases_initializer=bias_init)
    
    def policy(pre_layer):
        num_outputs=6
        activation_fn = tf.nn.softmax
        weights_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.1)
        return tf.contrib.layers.fully_connected(pre_layer,num_outputs=num_outputs,activation_fn=activation_fn,                                                    weights_initializer=weights_init, biases_initializer=bias_init)
    def value(pre_layer):
        num_outputs = 1
        activation_fn = None
        weights_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.1)
        
        return tf.contrib.layers.fully_connected(pre_layer,num_outputs=num_outputs,activation_fn=activation_fn,                                                weights_initializer=weights_init, biases_initializer=bias_init)


# In[ ]:


# --グローバルなTensorFlowのDeep Neural Networkのクラスです　-------
class ParameterServer:
    def __init__(self):
        with tf.variable_scope("parameter_server", reuse=tf.AUTO_REUSE):      # スレッド名で重み変数に名前を与え、識別します（Name Space）
            self._build_model()            # ニューラルネットワークの形を決定
            
        with tf.variable_scope("summary", reuse=tf.AUTO_REUSE):
            self._build_summary()

        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="parameter_server")
        self.train_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="parameter_server/trainable")
        self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, RMSProbDecaly)    # loss関数を最小化していくoptimizerの定義です
        
        self.saver = tf.train.Saver()
        
#         print("-------GLOBAL-------")
#         for w in self.weights_params:
#             print(w)

    def _build_model(self):
        self.state = NetworkSetting.state()            
        self.conv1 = NetworkSetting.conv1(self.state)
        self.maxpool1 = NetworkSetting.maxpool1(self.conv1)
        self.conv2 = NetworkSetting.conv2(self.maxpool1)
        self.maxpool2 = NetworkSetting.maxpool2(self.conv2)
        reshape = NetworkSetting.reshape(self.maxpool2)
        with tf.variable_scope("trainable"):
            fc1 = NetworkSetting.fc1(reshape)

            self.policy = NetworkSetting.policy(fc1)
            self.value = NetworkSetting.value(fc1)

        print("---------MODEL SHAPE-------------")
        print(self.state.get_shape())
        print(self.conv1.get_shape())
        print(self.conv2.get_shape())
        print(reshape.get_shape())
        print(fc1.get_shape())
        print(self.policy.get_shape())
        print(self.value.get_shape())
                
    def _build_summary(self):
        
        self.reward_ = tf.placeholder(tf.float32,shape=())
        self.loss_p_ = tf.placeholder(tf.float32, shape=())
        self.loss_v_ = tf.placeholder(tf.float32, shape=())
        
        self.summary_reward = tf.summary.merge([tf.summary.scalar('reward', self.reward_)])
        
        self.summary_loss = tf.summary.merge([tf.summary.scalar('loss_policy', self.loss_p_), tf.summary.scalar('loss_value', self.loss_v_)])
        
        conv1_display = tf.reshape(tf.transpose(self.conv1, [0,3,1,2]), (-1, self.conv1.get_shape()[1],self.conv1.get_shape()[2]))
        conv2_display = tf.reshape(tf.transpose(self.conv2, [0,3,1,2]), (-1, self.conv2.get_shape()[1],self.conv2.get_shape()[2]))
        conv1_display = tf.expand_dims(conv1_display, -1)
        conv2_display = tf.expand_dims(conv2_display, -1)
        
        state_shape = self.state.get_shape()
        conv1_shape = conv1_display.get_shape()
        conv2_shape = conv2_display.get_shape()

        s_img = []
        s_img.append(tf.summary.image('state',tf.reshape(self.state,[-1, state_shape[1], state_shape[2], state_shape[3]]), 1))
        s_img.append(tf.summary.image('conv1',tf.reshape(self.conv1,[-1, conv1_shape[1], conv1_shape[2], 1])))
        s_img.append(tf.summary.image('conv2',tf.reshape(self.conv2,[-1, conv2_shape[1], conv2_shape[2], 1])))
        
        self.summary_image = tf.summary.merge(s_img)
        self.writer = tf.summary.FileWriter("./logs",SESS.graph)
        
    def write_reward(self, step ,reward):
        m = SESS.run(self.summary_reward, feed_dict={self.reward_:r})
        return self.writer.add_summary(m, step)
    
    def write_loss(self, step, l_p, l_v):
        m = SESS.run(self.summary_reward, feed_dict={self.loss_p_: l_p, self.loss_v_:l_v})
        return self.writer.add_summary(m, step)
    
    def write_img(self, step, state):
        m = SESS.run(self.summary_image, feed_dict={self.state: state})
        return self.writer.add_summary(m, step)
        
    def save_model(self):
        self.saver.save(SESS, MODEL_PATH)
        
    def load_cnnweights(self, weights_path):
        assert len(weights_path) == 4
        cnn_weights = self.weights_params[:4]
        w_demo = [np.load(w_p) for w_p in weights_path]
        plh = [tf.placeholder(tf.float32, shape=w.shape) for w in w_demo]
        assign_op = [w.assign(p) for w, p in zip(cnn_weights, plh)]
        feed_dict = {p:w for w,p in zip(w_demo, plh)}
        SESS.run(assign_op, feed_dict)


# In[ ]:


class Agent(object):
    def __init__(self,name,network):
        self.name = name
        self.network = network
        self.memory = []
    
    def act(self,s1):
        
        global frames
        
        if frames>=EPS_STEPS:
            eps = EPS_END
        else:
            eps = EPS_START + frames*(EPS_END - EPS_START) / EPS_STEPS
        
        if random.random() < eps:
            action = np.zeros((N_ACTION,))
            action[np.random.randint(0,6)] = 1
            
            return action.tolist()
        else:
            s1 = np.array([s1])
            action_prob = self.network.predict_policy(s1)[0]
            
            action = np.zeros((N_ACTION,))
            action[np.random.choice(N_ACTION,p=action_prob)] = 1
#             action[np.argmax(action_prob)] = 1
            return action.tolist()
    
    def test_act(self,s1):
        s1 = np.array([s1])
        action_prob = self.network.predict_policy(s1)[0]

        action = np.zeros((N_ACTION,))
        action[np.random.choice(N_ACTION,p=action_prob)] = 1
#         action[np.argmax(action_prob)] = 1
        return action.tolist()
    
    def advantage_push_network(self,s1,action,reward,s2,isterminal):
        
        self.memory.append((s1,action,reward,s2))
        l_p = 0
        l_v = 0
        
        if isterminal:
            for i in range(len(self.memory)-1,-1,-1):
                s1,a,r,s2 = self.memory[i]
                if i==len(self.memory)-1:
                    self.R = 0
                else:
                    self.R = r + GAMMA*self.R
                
                self.network.train_push(s1,a,self.R,s2,isterminal)
            
            self.memory = []
            self.R = 0
            l_p, l_v = self.network.update_parameter_server()

        if len(self.memory)>=N_ADV:
            
            for i in range(N_ADV-1,-1,-1):
                s1,a,r,s2 = self.memory[i]
                if i==N_ADV-1:
                    self.R = self.network.predict_value(np.array([s1]))[0][0]
                else:
                    self.R = r + GAMMA*self.R
                
                self.network.train_push(s1,a,self.R,s2,isterminal)
            
            self.memory = []
            self.R = 0
            l_p, l_v = self.network.update_parameter_server()
            
        return l_p, l_v


# In[ ]:


class Network_local(object):
    def __init__(self,name,parameter_server):
        self.name = name
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self._model()
            self._build_graph(parameter_server)
            
        self.s1 = np.empty(shape=(100,RESOLUTION[0],RESOLUTION[1],RESOLUTION[2]),dtype=np.float32)
        self.s2 = np.empty(shape=(100,RESOLUTION[0],RESOLUTION[1],RESOLUTION[2]),dtype=np.float32)
        self.reward = np.empty(shape=(100,1),dtype=np.float32)
        self.action = np.empty(shape=(100,N_ACTION),dtype=np.float32)
        self.isterminal = np.empty(shape=(100,1),dtype=np.int8)
        self.queue_pointer = 0

#         print("-----LOCAL weights---")
#         for w in self.weights_params:
#             print(w)
            
#         print("-----LOCAL grads---")
#         for w in self.grads:
#             print(w)
    
    def _model(self):     # Kerasでネットワークの形を定義します
        
        self.state = NetworkSetting.state()
        self.conv1 = NetworkSetting.conv1(self.state)
        self.maxpool1 = NetworkSetting.maxpool1(self.conv1)
        self.conv2 = NetworkSetting.conv2(self.maxpool1)
        self.maxpool2 = NetworkSetting.maxpool2(self.conv2)
        reshape = NetworkSetting.reshape(self.maxpool2)
        
        with tf.variable_scope("trainable"):        
            fc1 = NetworkSetting.fc1(reshape)

            self.policy = NetworkSetting.policy(fc1)

            self.value = NetworkSetting.value(fc1)
            
    def _build_graph(self,parameter_server):
        
        self.a_t = tf.placeholder(tf.float32, shape=(None, N_ACTION))
        self.r_t = tf.placeholder(tf.float32, shape=(None,1))
        
        log_prob = tf.log(tf.reduce_sum(self.policy * self.a_t, axis=1, keep_dims=True)+1e-10)
        advantage = self.r_t - self.value
        
        self.loss_policy = -log_prob * tf.stop_gradient(advantage)
        self.loss_value = 0.5 * tf.square(advantage)
        entropy = 0.05 * tf.reduce_sum(self.policy * tf.log(self.policy + 1e-10), axis=1, keep_dims=True)
        self.loss_total = tf.reduce_mean(self.loss_policy + self.loss_value + entropy)
        
        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.train_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+"/trainable")
        self.grads = tf.gradients(self.loss_total, self.train_params)

        self.update_global_weight_params =             parameter_server.optimizer.apply_gradients(zip(self.grads, parameter_server.train_params))

        self.pull_global_weight_params = [l_p.assign(g_p) for l_p,g_p in zip(self.train_params,parameter_server.train_params)]

        self.push_local_weight_params = [g_p.assign(l_p) for g_p,l_p in zip(parameter_server.train_params,self.train_params)]
        
        self.pull_global_weights_params_all = [l_p.assign(g_p) for l_p,g_p in zip(self.weights_params,parameter_server.weights_params)]
        
    def pull_parameter_server(self):
        SESS.run(self.pull_global_weight_params)
    
    def push_parameter_server(self):
        SESS.run(self.push_local_weight_params)
        
    def show_weights(self):
        hoge = SESS.run(self.weights_params)
        for i in range(len(hoge)):
            print(hoge[i])
            
    def update_parameter_server(self):
        l_p = 0
        l_v = 0
        if self.queue_pointer > 0:
            s1 = self.s1[0:self.queue_pointer]
            s2 = self.s2[0:self.queue_pointer]
            r = self.reward[0:self.queue_pointer]
            a = self.action[0:self.queue_pointer]
            feed_dict = {self.state: s1,self.a_t:a, self.r_t:r}
            _, l_p, l_v = SESS.run([self.update_global_weight_params, self.loss_policy, self.loss_value],feed_dict)
#             print(hoge.shape)
#             print(hoge)
            self.queue_pointer = 0
        return l_p, l_v
    
    def predict_value(self,s):
        v = SESS.run(self.value,feed_dict={self.state:s})
        return v        
    
    def predict_policy(self,s):
        feed_dict = {self.state:s}
        prob = SESS.run(self.policy, feed_dict)
        return prob
    
    def train_push(self,s,a,r,s_,isterminal):
        self.s1[self.queue_pointer] = s
        self.s2[self.queue_pointer] = s_
        self.action[self.queue_pointer] = a
        self.reward[self.queue_pointer] = r
        self.isterminal[self.queue_pointer] = isterminal
        self.queue_pointer += 1
        
    def pull_all_parameter_server(self):
        SESS.run(self.pull_global_weights_params_all)     


# In[ ]:


# -- main ここからメイン関数です------------------------------
# M0.global変数の定義と、セッションの開始です
frames = 0              # 全スレッドで共有して使用する総ステップ数
isLearned = False       # 学習が終了したことを示すフラグ
config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list="0"))
config.log_device_placement = False
config.allow_soft_placement = True
SESS = tf.Session(config=config)
# SESS = tf_debug.LocalCLIDebugWrapperSession(SESS)

# スレッドを作成します
with tf.device("/gpu:1"):
    parameter_server = ParameterServer()    # 全スレッドで共有するパラメータを持つエンティティです
    parameter_server.load_cnnweights(MERGED_WEIGHTS_PATH)

    threads = []     # 並列して走るスレッド
    for i in range(N_WORKERS):
        thread_name = "local_thread"+str(i+1)
        threads.append(Worker_thread(thread_name=thread_name, parameter_server=parameter_server))

SESS.run(tf.global_variables_initializer())     # TensorFlowを使う場合、最初に変数初期化をして、実行します


# In[ ]:


test_env = TestEnvironment("test_env",parameter_server)

# TensorFlowでマルチスレッドを実行します
SESS.run(tf.global_variables_initializer())     # TensorFlowを使う場合、最初に変数初期化をして、実行します

# threads[0].environment.record=True

running_threads = []
start_time = time.time()
for worker in threads:
    job = lambda: worker.run()      # この辺は、マルチスレッドを走らせる作法だと思って良い
    t = threading.Thread(target=job)
    t.start()

test_frame = 0
while True:
    
    if frames >= test_frame and frames<test_frame+1000:
        test_env.run()
        test_frame += 1000
    elif frames >= test_frame+1000:
        print("TEST at %d~%d step cant be finished"%(test_frame, test_frame+1000-1))
        test_frame += 1000
    else:
        pass
    
    isLearned = True
    for worker in threads:
        if not worker.environment.finished:
            isLearned = False
    
    if isLearned:
        break

print("*****************************\nTIME to LEARNING:%.3f [sec]\n*****************************"%(time.time()-start_time))

# np.save("./records/reward.npy",np.array(test_env.record_reward))
# np.save("./records/frag.npy",np.array(test_env.record_frag))
# np.save("./records/death.npy",np.array(test_env.record_death))
# np.save("./records/loss_policy.npy",threads[0].environment.loss_policy)
# np.save("./records/loss_value.npy",threads[0].environment.loss_value)
# np.save("./records/entropy.npy",threads[0].environment.entropy)
# np.save("./records/frame.npy",threads[0].environment.frame_record)

# parameter_server.save_model()
# print("Learning phase is finished")
for i in range(3):
    test_env.run()


# In[ ]:




