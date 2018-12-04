
# coding: utf-8

# In[ ]:


#!/usr/bin/python

from __future__ import print_function
import math
import time,random,threading,datetime
import tensorflow as tf
from time import sleep
import numpy as np
from vizdoom import *
import skimage.color, skimage.transform
from tqdm import tqdm
import h5py
from tensorflow.python import debug as tf_debug
from tqdm import tqdm
from PIL import Image

CONFIG_FILE_PATH = "./config/custom_config.cfg"
MODEL_PATH = "./model_v03/model_v03.ckpt"
DEMO_PATH = "./demodata_cig2017_v0-2.h5"
LOG_DIR = "./logs_v03"
RECORDS_DIR = "./records_v03/"
N_FOLDER = 3

RESOLUTION = (120,180,3)

N_ADV = 10

TIME_LEARN = datetime.timedelta(hours=24)

N_WORKERS = 10

WORKER_STEPS = 200000

N_PRE_STEP = 20000

N_ACTION = 6

GAMMA = 0.99

BOTS_NUM = 10

EPS_START = 0.5
EPS_END = 0.0
EPS_STEPS = WORKER_STEPS*N_WORKERS

REWARDS = {'living':-0.01, 'health_loss':-1, 'medkit':50, 'ammo':0.0, 'frag':500, 'dist':3e-2, 'suicide':-500} 

LEARNING_RATE = 5e-3
RMSProbDecaly = 0.99

ISNPY = False

TEST_FREQ_PRELEARNING = 50

COEFF_L_P = 1.0
COEFF_L_V = 0.01
COEFF_ENT = 1.0


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
        global frames,runout
        
        self.start_episode()
        
        train_episode = 0
        step = 0
#         for step in range(WORKER_STEPS):
        while True:
            
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
                    self.parameter_server.write_summary(frames,np.array([s1]),np.array([action]),np.array([[reward]]))
                    l_p,l_v,ent = self.parameter_server.calc_loss(frames,np.array([s1]),np.array([action]),np.array([[reward]]))
                    self.loss_policy.append(l_p[0][0])
                    self.loss_value.append(l_v[0][0])
                    self.entropy.append(ent[0][0])
                    self.frame_record.append(frames)
                    
#                 if step==0:
#                     print("async")
#                     print(s1.shape)
#                     print(action)
#                     print(reward)
                    

                self.agent.advantage_push_network(s1,action,reward,s2,isterminal)
                
                if self.game.is_player_dead():
                    self.game.respawn_player()
                    self.reward_gen.respawn_pos(self.game.get_game_variable(GameVariable.HEALTH),                                                 self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO),                                                 self.game.get_game_variable(GameVariable.POSITION_X),                                                self.game.get_game_variable(GameVariable.POSITION_Y))

            else:
                train_episode += 1
                self.start_episode()
                self.reward_gen.new_episode(health = self.game.get_game_variable(GameVariable.HEALTH),                                            ammo = self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO),                                            posx = self.game.get_game_variable(GameVariable.POSITION_X),                                            posy = self.game.get_game_variable(GameVariable.POSITION_Y))
            self.local_step += 1   
            frames += 1
            step += 1
            
            if runout == True:
                break
                
        print(self.name," finished")
        self.finished = True


# In[ ]:


class DemoEnvironment(object):
    def __init__(self,name, parameter_server, test_env):
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
        
        self.test_env = test_env
        
        self.pre_death = 0
        
        self.parameter_server = parameter_server
        
        self.replay_buff = []
        
        self.loss_policy = []
        self.loss_value = []
        self.entropy = []
        self.frame_record = []

    def load_demonstration(self):
        
        if ISNPY:
            pass
        else:
            hdf5file = h5py.File(DEMO_PATH,"r")
            folder = "demodata_"+str(0)
            state1 = hdf5file[folder+"/state1"].value
            state2 = hdf5file[folder+"/state2"].value
            actions_row = hdf5file[folder+"/actions"].value
            isterminals = hdf5file[folder+"/isterminals"].value
            health = hdf5file[folder+"/healths"].value
            ammo = hdf5file[folder+"/ammos"].value
            posx = hdf5file[folder+"/posxs"].value
            posy = hdf5file[folder+"/posys"].value
            death = hdf5file[folder+"/deaths"].value
            frag = hdf5file[folder+"/frags"].value
            
            for i in range(1,N_FOLDER):
                folder = "demodata_" +str(i)
                state1 = np.concatenate((state1,hdf5file[folder+"/state1"].value),axis=0)
                state2 = np.concatenate((state2,hdf5file[folder+"/state2"].value),axis=0)
                actions_row = np.concatenate((actions_row,hdf5file[folder+"/actions"].value),axis=0)
                isterminals = np.concatenate((isterminals,hdf5file[folder+"/isterminals"].value),axis=0)
                health = np.concatenate((health,hdf5file[folder+"/healths"].value),axis=0)
                ammo = np.concatenate((ammo,hdf5file[folder+"/ammos"].value),axis=0)
                posx = np.concatenate((posx,hdf5file[folder+"/posxs"].value),axis=0)
                posy = np.concatenate((posy,hdf5file[folder+"/posys"].value),axis=0)
                death = np.concatenate((death,hdf5file[folder+"/deaths"].value),axis=0)
                frag = np.concatenate((frag,hdf5file[folder+"/frags"].value),axis=0)
            
            n_transit, n_step, _ = actions_row.shape
            print(actions_row.shape)
            
            actions = np.zeros(shape=[n_transit,n_step,N_ACTION])
            for i in range(n_transit):
                for j in range(n_step):
                    actions[i][j][actions_row[i][j]] = 1
                    

            is_dead = False
            is_finished = False

            pre_health = 100
            pre_ammo = 15
            pre_frag = 0
            pre_death = 0
            pre_posx = 0.0
            pre_posy = 0.0


            for i in range(n_transit):

                pre_posx = posx[i][0]
                pre_posy = posy[i][0]
                
                replay_sequence = []
                R = 0

                for j in range(N_ADV):
                    if not is_finished:
                        if is_dead :
                            pre_posx = posx[i][j]
                            pre_posy = posy[i][j]
                            pre_health = 100
                            pre_ammo = 15
                            is_dead = False

                        m_frag = frag[i][j] - pre_frag
                        m_death = death[i][j] - pre_death
                        m_health = health[i][j] - pre_health
                        m_ammo = ammo[i][j] - pre_ammo
                        m_posx = posx[i][j] - pre_posx
                        m_posy = posy[i][j] - pre_posy

                        if m_death >= 1:
                            is_dead = True 

                        if isterminals[i][j] == True:
                            is_finished = True
                            
                        r_d = self.reward_gen.calc_reward(m_frag,m_death,m_health,m_ammo,m_posx,m_posy)
                        r = sum(r_d.values())
                        replay_sequence.append((state1[i][j],actions[i][j],state2[i][j],r,isterminals[i][j]))

                        pre_frag = frag[i][j]
                        pre_death = death[i][j]
                        pre_health = health[i][j]
                        pre_ammo = ammo[i][j]

#                         if j==7:
#                             print("---------------------------------")
#                             print("state[%d][%d]:"%(i,j))
#                             print("\t%4.2f, %4.2f, %4.2f, %4.2f, %4.2f, %4.2f"%(m_frag,m_death,m_health,m_ammo,m_posx,m_posy))
#                             print("\t%4.2f, %4.2f, %4.2f, %4.2f, %4.2f, %4.2f"%(r_d['frag'],r_d['medkit'],r_d['health_loss'],r_d['ammo'],r_d['dist'],r_d['suicide']))
                    else:
                        replay_sequence.append((None,None,None,None,True))
                        
                is_finished = False
                
                self.replay_buff.append(replay_sequence)
                replay_sequence = []
            
    def pre_learning(self):

        global frames
        num_replay = len(self.replay_buff)

        for step in tqdm(range(N_PRE_STEP)):
            self.agent.network.pull_parameter_server()
            replay_idx = random.randint(0,num_replay-1)
            
            if frames%TEST_FREQ_PRELEARNING == 0:
                self.test_env.run()

            for i in range(N_ADV):
                s1 = self.replay_buff[replay_idx][i][0]
                action = self.replay_buff[replay_idx][i][1]
                s2 = self.replay_buff[replay_idx][i][2]
                reward = self.replay_buff[replay_idx][i][3]
                isterminal = self.replay_buff[replay_idx][i][4]
                if not(type(s1)==type(None)):
                    self.agent.advantage_push_network(s1,action,reward,s2,isterminal)
                
            s1 = self.replay_buff[replay_idx][0][0]
            action = self.replay_buff[replay_idx][0][1]
            reward = self.replay_buff[replay_idx][0][3]
            
#             print("-----------%d---------------"%step)
#             print("-----------------GLOBAL-----------------")
#             self.parameter_server.test_func(s1,action,reward)
#             print("-----------------LOCAL-----------------")
#             self.agent.network.test_func(s1,action,reward)

            l_p,l_v,ent = self.parameter_server.calc_loss(frames,np.array([s1]),np.array([action]),np.array([reward]))
            self.loss_policy.append(l_p[0][0])
            self.loss_value.append(l_v[0][0])
            self.entropy.append(ent[0][0])
            self.frame_record.append(frames)
            
            frames+=1
            


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
            action = self.agent.test_act(s1)
            self.game.make_action(action,1)
            reward = self.get_reward()
            isterminal = self.game.is_episode_finished()

            if self.game.is_player_dead():
                self.game.respawn_player()
                self.reward_gen.respawn_pos(self.game.get_game_variable(GameVariable.HEALTH),                                             self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO),                                             self.game.get_game_variable(GameVariable.POSITION_X),                                            self.game.get_game_variable(GameVariable.POSITION_Y))
            
            step += 1
                
        print("----------TEST at %d step-------------"%(frames))
        ret_frag = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        ret_death = self.game.get_game_variable(GameVariable.DEATHCOUNT)-self.pre_death
        ret_reward = self.reward_gen.total_reward
        print("FRAG:",ret_frag,"DEATH:",ret_death)
        print("REWARD",ret_reward)
        print("DETAIL:",self.reward_gen.total_reward_detail)
        return ret_reward,ret_frag,ret_death
#         self.record_frag.append(self.game.get_game_variable(GameVariable.FRAGCOUNT))
#         self.record_death.append(self.game.get_game_variable(GameVariable.DEATHCOUNT)-self.pre_death)
#         self.record_reward.append(self.reward_gen.total_reward)
        self.pre_death = self.game.get_game_variable(GameVariable.DEATHCOUNT)
        
    def run_gif(self):
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
        gif_img = []
        while not self.game.is_episode_finished():
            
            if step%N_ADV==0 and not step==0:
                self.reward_gen.update_origin(self.game.get_game_variable(GameVariable.POSITION_X),                                              self.game.get_game_variable(GameVariable.POSITION_Y))

            s1 = self.preprocess(self.game.get_state().screen_buffer)
            if step%5==0:
                gif_img.append(s1)
            action = self.agent.test_act(s1)
            self.game.make_action(action,1)
            reward = self.get_reward()
            isterminal = self.game.is_episode_finished()

            if self.game.is_player_dead():
                self.game.respawn_player()
                self.reward_gen.respawn_pos(self.game.get_game_variable(GameVariable.HEALTH),                                             self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO),                                             self.game.get_game_variable(GameVariable.POSITION_X),                                            self.game.get_game_variable(GameVariable.POSITION_Y))
            
            step += 1
        
        save_img = []
        for i in range(len(gif_img)):
            save_img.append(Image.fromarray(np.uint8(gif_img[i]*255)))
        
        save_img[0].save('test.gif',save_all=True,append_images=save_img[1:])
                
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
        
#         with tf.device("/cpu:0"):
#             self.saver = tf.train.Saver()
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
        
        self.reward = tf.placeholder(tf.float32,shape=(None,1))
        self.frag = tf.placeholder(tf.int64,shape=(None,1))
        self.death = tf.placeholder(tf.int64,shape=(None,1))
        
        log_prob = tf.log(tf.reduce_sum(self.policy * self.a_t, axis=1, keep_dims=True)+1e-10)
        advantage = self.r_t - self.value
        
        self.test1 = log_prob
        self.test2 = advantage
        
        self.loss_policy = -log_prob * tf.stop_gradient(advantage)
        self.loss_value = COEFF_L_V* tf.square(advantage)
        self.entropy = COEFF_ENT * tf.reduce_sum(self.policy * tf.log(self.policy + 1e-10), axis=1, keep_dims=True)
        
        print(self.loss_policy.get_shape())
        print(self.loss_value.get_shape())
        
#         with tf.device("/cpu:0"):
        summary_lp = tf.summary.scalar('loss_policy',self.loss_policy[0][0])
        summary_lv = tf.summary.scalar('loss_value', self.loss_value[0][0])
        summary_ent = tf.summary.scalar('entropy', self.entropy[0][0])

        self.merged_scalar = tf.summary.merge([summary_lp,summary_lv,summary_ent])

        state_shape = self.state.get_shape()
        conv1_shape = self.conv1.get_shape()
        conv2_shape = self.conv2.get_shape()
        summary_state  = tf.summary.image('state',tf.reshape(self.state,[-1, state_shape[1], state_shape[2], state_shape[3]]),1)
        summary_conv1 = tf.summary.image('conv1',tf.reshape(self.conv1,[-1, conv1_shape[1], conv1_shape[2], 1]),1)
        summary_conv2 = tf.summary.image('conv2',tf.reshape(self.conv2,[-1, conv2_shape[1], conv2_shape[2], 1]),1)

        self.merged_image = tf.summary.merge([summary_state,summary_conv1,summary_conv2])
        
        summary_reward = tf.summary.scalar('reward',self.reward[0][0])
        summary_frag = tf.summary.scalar('frag',self.frag[0][0])
        summary_death = tf.summary.scalar('death',self.death[0][0])
        
        self.merged_testscore = tf.summary.merge([summary_reward,summary_frag,summary_death])
        
        self.writer = tf.summary.FileWriter(LOG_DIR,SESS.graph)
        
    def test_func(self,s1,a,r):
        s1 = np.array([s1])
        a = np.array([a])
        r = np.array([r])
        
        temp=SESS.run([self.value,self.test1,self.test2,self.loss_policy,self.loss_value,self.entropy],feed_dict={self.state:s1, self.a_t:a, self.r_t:r})
        
        print("\tvalue:%f\n\ttest1:%f\n\ttest2:%f\n\tloss_policy:%f\n\tloss_value:%f"%(temp[0],temp[1],temp[2],temp[3],temp[4]))
        
    def calc_loss(self,step,s1,a,r):
        loss_p,loss_v,entropy = SESS.run([self.loss_policy,self.loss_value,self.entropy],feed_dict={self.state:s1,self.a_t:a,self.r_t:r})
        return loss_p,loss_v,entropy
        
    def write_summary(self,step,s1,a,r):
        m_s,m_i = SESS.run([self.merged_scalar,self.merged_image],feed_dict={self.state:s1,self.a_t:a,self.r_t:r})
        self.writer.add_summary(m_s,step)
        if step%1000 == 0:
            self.writer.add_summary(m_i,step)
    
    def write_records(self,step,r,f,d):
        r = np.array([[r]])
        f = np.array([[f]])
        d = np.array([[d]])
        m = SESS.run(self.merged_testscore,feed_dict={self.reward:r,self.frag:f,self.death:d})
        self.writer.add_summary(m,step)
        
    def save_model(self):
        self.saver.save(SESS, MODEL_PATH)


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
            self.network.update_parameter_server()

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
            self.network.update_parameter_server()
    


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
        
        self.test1 = log_prob
        self.test2 = advantage

        loss_policy = -log_prob * tf.stop_gradient(advantage)
        loss_value = COEFF_L_V * tf.square(advantage)
        entropy = COEFF_ENT *  tf.reduce_sum(self.policy * tf.log(self.policy + 1e-10), axis=1, keep_dims=True)
        self.loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)
        
        self.test3 = loss_policy
        self.test4 = loss_value
        self.test5 = entropy
        
        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.grads = tf.gradients(self.loss_total, self.weights_params)

#         for i in self.grads:
#             print(i)
            
#         for i in self.weights_params:
#             print(i)

        self.update_global_weight_params =             parameter_server.optimizer.apply_gradients(zip(self.grads, parameter_server.weights_params))

        self.pull_global_weight_params = [l_p.assign(g_p) for l_p,g_p in zip(self.weights_params,parameter_server.weights_params)]

        self.push_local_weight_params = [g_p.assign(l_p) for g_p,l_p in zip(parameter_server.weights_params,self.weights_params)]
    
    def test_func(self,s1,a,r):
        s1 = np.array([s1])
        a = np.array([a])
        r = np.array([r])
#         temp=SESS.run([self.value,self.test1,self.test2,self.test3,self.test4,self.test5],feed_dict={self.state:s1, self.a_t:a, self.r_t:r})
#         print("\tvalue:%f\n\ttest1:%f\n\ttest2:%f\n\tloss_policy:%f\n\tloss_value:%f"%(temp[0],temp[1],temp[2],temp[3],temp[4]))
        temp = SESS.run([self.grads],feed_dict={self.state:s1, self.a_t:a, self.r_t:r})
#         print(temp)
    
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
#             print(self.name,s1.shape)
#             print(self.queue_pointer)
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


# -- main ここからメイン関数です------------------------------
# M0.global変数の定義と、セッションの開始です
frames = 0              # 全スレッドで共有して使用する総ステップ数
pre_frames = 0
isLearned = False       # 学習が終了したことを示すフラグ
runout = False
SESS = tf.Session(config=tf.ConfigProto(log_device_placement=False))     # TensorFlowのセッション開始
# SESS = tf_debug.LocalCLIDebugWrapperSession(SESS)

# スレッドを作成します
with tf.device("/cpu:0"):
    parameter_server = ParameterServer()    # 全スレッドで共有するパラメータを持つエンティティです
with tf.device("/gpu:0"):
    threads = []     # 並列して走るスレッド
    for i in range(N_WORKERS):
        thread_name = "local_thread"+str(i+1)
        threads.append(Worker_thread(thread_name=thread_name, parameter_server=parameter_server))

    test_env = TestEnvironment("test_env",parameter_server)
    prelearning_env = DemoEnvironment("pre_learning",parameter_server,test_env)

# TensorFlowでマルチスレッドを実行します
SESS.run(tf.global_variables_initializer())     # TensorFlowを使う場合、最初に変数初期化をして、実行します

threads[0].environment.record=True

print("loading_demonstration")
prelearning_env.load_demonstration()
start_time = datetime.datetime.now()
print("pre learning")
prelearning_env.pre_learning()

# np.save(RECORDS_DIR+"pre_reward.npy",np.array(test_env.record_reward))
# np.save(RECORDS_DIR+"pre_frag.npy",np.array(test_env.record_frag))
# np.save(RECORDS_DIR+"pre_death.npy",np.array(test_env.record_death))
# np.save(RECORDS_DIR+"pre_loss_policy.npy",prelearning_env.loss_policy)
# np.save(RECORDS_DIR+"pre_loss_value.npy",prelearning_env.loss_value)
# np.save(RECORDS_DIR+"pre_entropy.npy",prelearning_env.entropy)
# np.save(RECORDS_DIR+"pre_frame.npy",prelearning_env.frame_record)

test_env.record_reward = []
test_env.record_frag = []
test_env.record_death = []
frames = 0

sleep(3.0)

start_time_async = datetime.datetime.now()

running_threads = []
for worker in threads:
    job = lambda: worker.run()      # この辺は、マルチスレッドを走らせる作法だと思って良い
    t = threading.Thread(target=job)
    t.start()

test_frame = 0
while True:
    
    if frames >= test_frame and frames<test_frame+1000:
        r,f,d = test_env.run()
        parameter_server.write_records(frames,r,f,d)
        test_frame += 1000
    elif frames >= test_frame+1000:
        print("TEST at %d~%d step cant be finished"%(test_frame, test_frame+1000-1))
        test_frame += 1000
    else:
        pass
    
    if datetime.datetime.now() > start_time_async+TIME_LEARN:
        runout=True
    
    isLearned = True
    for worker in threads:
        if not worker.environment.finished:
            isLearned = False
    
    if isLearned:
        break

print("*****************************\nTIME to LEARNING:%.3f [sec]\n*****************************"%(datetime.datetime.now()-start_time).seconds)

# np.save(RECORDS_DIR+"reward.npy",np.array(test_env.record_reward))
# np.save(RECORDS_DIR+"frag.npy",np.array(test_env.record_frag))
# np.save(RECORDS_DIR+"death.npy",np.array(test_env.record_death))
# np.save(RECORDS_DIR+"loss_policy.npy",threads[0].environment.loss_policy)
# np.save(RECORDS_DIR+"loss_value.npy",threads[0].environment.loss_value)
# np.save(RECORDS_DIR+"entropy.npy",threads[0].environment.entropy)
# np.save(RECORDS_DIR+"frame.npy",threads[0].environment.frame_record)

parameter_server.save_model()
print("Learning phase is finished")
for i in range(3):
    test_env.run()

test_env.run_gif()

