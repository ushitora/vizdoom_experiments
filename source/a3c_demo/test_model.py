
# coding: utf-8

# In[ ]:


#!/usr/bin/python

from __future__ import print_function
import math
import time,random,threading
import tensorflow as tf
from time import sleep
import numpy as np
from vizdoom import *
import skimage.color, skimage.transform
from tqdm import tqdm
from tensorflow.python import debug as tf_debug

CONFIG_FILE_PATH = "./config/custom_config.cfg"
MODEL_PATH = "./model_v02/model_v02.ckpt"
RESOLUTION = (120,180,3)

N_ADV = 5

UPDATE_FREQ = 10

N_ACTION = 6

GAMMA = 0.99

BOTS_NUM = 5

REWARDS = {'living':-0.01, 'health_loss':-1, 'medkit':50, 'ammo':0.0, 'frag':500, 'dist':3e-2, 'suicide':-500} 

LEARNING_RATE = 5e-3
RMSProbDecaly = 0.99


# In[ ]:


class TestEnvironment(object):
    def __init__(self,name, parameter_server):
        self.game = DoomGame()
        self.game.load_config(CONFIG_FILE_PATH)
        self.game.set_window_visible(True)
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
        self.reward_gen = RewardGenerater(health,ammo,frag,pos_x,pos_y)
        
        self.start_episode()
        
        #Copy params from global
        self.agent.network.pull_parameter_server()

        step = 0
        while not self.game.is_episode_finished():
            
            if step%N_ADV==0 and not step==0:
                self.reward_gen.update_origin(self.game.get_game_variable(GameVariable.POSITION_X),                                              self.game.get_game_variable(GameVariable.POSITION_Y))

            s1 = self.preprocess(self.game.get_state().screen_buffer)
            action = self.agent.test_act(s1)
            value = self.agent.network.predict_value(np.array([s1]))
            self.game.make_action(action,1)
            reward = self.get_reward()
            isterminal = self.game.is_episode_finished()
            
            print("#",step,":Action:",action,"->Reward:",reward)
            print("value:",value)

            if self.game.is_player_dead():
                self.game.respawn_player()
                self.reward_gen.respawn_pos(self.game.get_game_variable(GameVariable.HEALTH),                                             self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO),                                             self.game.get_game_variable(GameVariable.POSITION_X),                                            self.game.get_game_variable(GameVariable.POSITION_Y))
            
            step += 1
                
        print("----------TEST at %d step-------------"%(frames))
        print("FRAG:",self.game.get_game_variable(GameVariable.FRAGCOUNT),"DEATH:",self.game.get_game_variable(GameVariable.DEATHCOUNT)-self.pre_death)
        print("REWARD",self.reward_gen.total_reward)
        print("DETAIL:",self.reward_gen.total_reward_detail)
        self.pre_death = self.game.get_game_variable(GameVariable.DEATHCOUNT)


# In[ ]:


class NetworkSetting:
    
    def state():
        name = "STATE"
        shape = [None,RESOLUTION[0],RESOLUTION[1],RESOLUTION[2]]
        return tf.placeholder(tf.float32,shape=shape,name=name)
    
    def conv1(pre_layer):
        num_outputs = 8
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
        num_outputs = 16
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
        with tf.variable_scope("parameter_server"):      # スレッド名で重み変数に名前を与え、識別します（Name Space）
            self._build_model()            # ニューラルネットワークの形を決定
            
        with tf.variable_scope("summary"):
            self._build_summary()

        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="parameter_server")
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
            fc1 = NetworkSetting.fc1(reshape)

            with tf.variable_scope("policy"):
                self.policy = NetworkSetting.policy(fc1)
            
            with tf.variable_scope("value"):
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
        
        self.a_t = tf.placeholder(tf.float32, shape=(None, N_ACTION))
        self.r_t = tf.placeholder(tf.float32, shape=(None,1))
        
        log_prob = tf.log(tf.reduce_sum(self.policy * self.a_t, axis=1, keep_dims=True)+1e-10)
        advantage = self.r_t - self.value
        
        loss_policy = log_prob * tf.stop_gradient(advantage)
        loss_value = 0.5 * tf.square(advantage)
        entropy = 0.05 * tf.reduce_sum(self.policy * tf.log(self.policy + 1e-10), axis=1, keep_dims=True)
        
        print(loss_policy.get_shape())
        print(loss_value.get_shape())
        
        tf.summary.scalar('loss_policy',loss_policy[0][0])
        tf.summary.scalar('loss_value', loss_value[0][0])
        tf.summary.scalar('entropy', entropy[0][0])
        
        state_shape = self.state.get_shape()
        conv1_shape = self.conv1.get_shape()
        conv2_shape = self.conv2.get_shape()
        tf.summary.image('state',tf.reshape(self.state,[-1, state_shape[1], state_shape[2], state_shape[3]]),1)
        tf.summary.image('conv1',tf.reshape(self.conv1,[-1, conv1_shape[1], conv1_shape[2], 1]),1)
        tf.summary.image('conv2',tf.reshape(self.conv2,[-1, conv2_shape[1], conv2_shape[2], 1]),1)
        
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("./logs",SESS.graph)
        
    def write_summary(self,step,s1,a,r):
        m = SESS.run(self.merged,feed_dict={self.state:s1,self.a_t:a,self.r_t:r})
        self.writer.add_summary(m,step)
        
    def save_model(self):
        self.saver.save(SESS, MODEL_PATH)
        
    def load_model(self):
        self.saver.restore(SESS,MODEL_PATH)


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


class Network_local(object):
    def __init__(self,name,parameter_server):
        self.name = name
        with tf.variable_scope(self.name):
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
        fc1 = NetworkSetting.fc1(reshape)

        with tf.variable_scope("policy"):
            self.policy = NetworkSetting.policy(fc1)

        with tf.variable_scope("value"):
            self.value = NetworkSetting.value(fc1)
            
    def _build_graph(self,parameter_server):
        
        self.a_t = tf.placeholder(tf.float32, shape=(None, N_ACTION))
        self.r_t = tf.placeholder(tf.float32, shape=(None,1))
        
        log_prob = tf.log(tf.reduce_sum(self.policy * self.a_t, axis=1, keep_dims=True)+1e-10)
        advantage = self.r_t - self.value
        
        loss_policy = -log_prob * tf.stop_gradient(advantage)
        loss_value = 0.5 * tf.square(advantage)
        entropy = 0.05 * tf.reduce_sum(self.policy * tf.log(self.policy + 1e-10), axis=1, keep_dims=True)
        self.loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)
        
        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.grads = tf.gradients(self.loss_total, self.weights_params)

        self.update_global_weight_params =             parameter_server.optimizer.apply_gradients(zip(self.grads, parameter_server.weights_params))

        self.pull_global_weight_params = [l_p.assign(g_p) for l_p,g_p in zip(self.weights_params,parameter_server.weights_params)]

        self.push_local_weight_params = [g_p.assign(l_p) for g_p,l_p in zip(parameter_server.weights_params,self.weights_params)]
        
    def pull_parameter_server(self):
        SESS.run(self.pull_global_weight_params)
    
    def push_parameter_server(self):
        SESS.run(self.push_local_weight_params)
        
    def show_weights(self):
        hoge = SESS.run(self.weights_params)
        for i in range(len(hoge)):
            print(hoge[i])
            
    def update_parameter_server(self):
        if self.queue_pointer > 0:
            s1 = self.s1[0:self.queue_pointer]
            s2 = self.s2[0:self.queue_pointer]
            r = self.reward[0:self.queue_pointer]
            a = self.action[0:self.queue_pointer]
            feed_dict = {self.state: s1,self.a_t:a, self.r_t:r}
            SESS.run(self.update_global_weight_params,feed_dict)
            self.queue_pointer = 0
    
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


# In[ ]:


class Agent(object):
    def __init__(self,name,network):
        self.name = name
        self.network = network
    
    def test_act(self,s1):
        s1 = np.array([s1])
        action_prob = self.network.predict_policy(s1)[0]

        action = np.zeros((N_ACTION,))
#         action[np.argmax(action_prob)] = 1
#         print(action_prob)
        action[np.random.choice(6,p=action_prob)] = 1
        return action.tolist()    


# In[ ]:


# -- main ここからメイン関数です------------------------------
# M0.global変数の定義と、セッションの開始です
frames = 0              # 全スレッドで共有して使用する総ステップ数
isLearned = False       # 学習が終了したことを示すフラグ
SESS = tf.Session()     # TensorFlowのセッション開始
# SESS = tf_debug.LocalCLIDebugWrapperSession(SESS)
TRAINING = "training"
TESTING = "testing"

# M1.スレッドを作成します
with tf.device("/cpu:0"):
    parameter_server = ParameterServer()    # 全スレッドで共有するパラメータを持つエンティティです
    parameter_server.load_model()

test_env = TestEnvironment("test_env",parameter_server)

# M2.TensorFlowでマルチスレッドを実行します
COORD = tf.train.Coordinator()                  # TensorFlowでマルチスレッドにするための準備です
SESS.run(tf.global_variables_initializer())     # TensorFlowを使う場合、最初に変数初期化をして、実行します

test_env.run()
test_env.game.close()

