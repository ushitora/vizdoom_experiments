#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import multiprocessing
import skimage.color, skimage.transform
from vizdoom import *
import os, time, random, threading, h5py, math,pickle
import tensorflow as tf
import numpy as np
from game_instance_basic import GameInstanceBasic, GameInstanceSimpleBasic
from global_constants import *
from datetime import datetime, timedelta, timezone
from PIL import Image
import matplotlib.pyplot as plt
from replay_memory import ReplayMemory
import pandas as pd
# %matplotlib inline


# In[ ]:


JST = timezone(timedelta(hours=+9),'JST')

DATETIME = datetime.now(JST)
LOGDIR = "../data/demo_dqn/logs/log_"+DATETIME.strftime("%Y-%m-%d-%H-%M-%S")+"/"
MODEL_PATH =  "../data/demo_dqn/models/model_"+DATETIME.strftime("%Y-%m-%d-%H-%M-%S")+"/model.ckpt"
# CONFIG_FILE_PATH = "./config/simple_deathmatch.cfg"
# CONFIG_FILE_PATH = "./config/simple_deathmatch_h40_s20.cfg"
# CONFIG_FILE_PATH = "./config/simple_deathmatch_h20_s0_noattack_nearspawn.cfg"
# CONFIG_FILE_PATH = "./config/large_basic_randomspawn.cfg"
# CONFIG_FILE_PATH = "./config/simpler_basic.cfg"
# CONFIG_FILE_PATH = "./config/large_basic_pistol.cfg"
# CONFIG_FILE_PATH = "./config/large_basic_pistol_hate.cfg"
CONFIG_FILE_PATH = "./config/large_basic_rocket_hate.cfg"
PLAY_LOGDIR = "../data/demo_dqn/playlogs/playlog_"+DATETIME.strftime("%Y-%m-%d-%H-%M-%S")+"/"
# DEMO_PATH = ["../demonstration/large_basic/demo_largebasic01.hdf5","../demonstration/large_basic_randomspawn/demo_largebasic_randomspawn.hdf5"]
# DEMO_PATH = ["../demonstration/large_basic_pistol/demo_largebasic_pistol01.hdf5"]
# DEMO_PATH = ["../demonstration/large_basic_pistol_6action/demo_largebasic_pistol_6action.hdf5"]
# DEMO_PATH = ["../demonstration/large_basic_rocket_6action/demo_largebasic_rocket_6action01.hdf5"]
# DEMO_PATH = ["../demonstration/largebasic_pistol_hate/demo_largebasic_pistol_hate01.hdf5"]
DEMO_PATH = ["../demonstration/largebasic_rocket_hate/demo_largebasic_rocket_hate01.hdf5"]
# POSITIVEDATA_PATH = ["../data/positive_replay/positive_replay05.npy","../data/positive_replay/positive_replay06.npy"]
# __name__ = "learning_imitation"
__name__ = "learning_async"
# __name__ = "test"
N_ACTION = 6
N_AGENT_ACTION = 2**6
BOTS_NUM = 1
N_WORKERS = 10
# REWARDS = {'living':-0.01, 'healthloss':-0.01, 'medkit':0.0, 'ammo':0.0, 'frag':1.0, 'dist':1e-4, 'suicide':-1.0}
# REWARDS = {'living':-1.0, 'healthloss':0.0, 'medkit':0.0, 'ammo':0.0, 'frag':100.0, 'dist':0.0, 'suicide':-100.0, 'kill':100.0,'death':-100.0,'enemysight':0.0, 'ammoloss':0.0}
REWARDS = {'living':-1.0, 'healthloss':0.0, 'medkit':0.0, 'ammo':0.0, 'frag':0.0, 'dist':0.0, 'suicide':0.0, 'kill':100.0,'death':-100.0,'enemysight':0.0, 'ammoloss':0.0}
# REWARDS = {'living':-0.01, 'healthloss':0.0, 'medkit':0.0, 'ammo':0.0, 'frag':0.0, 'dist':0.0, 'suicide':-1.0, 'kill':1.0,'death':-1.0,'enemysight':0.0, 'ammoloss':0.0}
LSTM_SIZE = 1024
N_ADV = 5
N_SEQ = 5
LAMBDA_ONE = 1.0
LAMBDA1 = 1.0
LAMBDA2 = 1.0
# LAMBDA3 = 0.00001
LAMBDA3 = 0.0001
RESOLUTION = (120,120,3)
MERGIN_VALUE = 0.02
INTERVAL_BATCH_LEARNING = 10
INTERVAL_UPDATE_NETWORK = 10
INTERVAL_PULL_PARAMS = 1
N_BATCH = 64
INTERVAL_UPDATE_ORIGIN = 10
USED_GPU = "0"
BETA_MIN = 0.0
BETA_MAX = 0.4
EPS_MAX = 0.9
EPS_MIN = 0.5
N_STEPS = 30000
IMIT_MODEL_PATH = "../data/demo_dqn/models/model_2019-02-01-03-28-37/model.ckpt"


# In[ ]:


if not os.path.exists(LOGDIR):
    os.mkdir(LOGDIR)
if not os.path.exists(os.path.dirname(MODEL_PATH)):
    os.mkdir(os.path.dirname(MODEL_PATH))
if not os.path.exists(PLAY_LOGDIR):
    os.mkdir(PLAY_LOGDIR)


# In[ ]:


class Environment(object):
    def __init__(self,sess,  name, game_instance, network, agent, start_time=None, end_time=None, n_step=None,  random_seed=0):
#     def __init__(self,sess,  name, start_time, end_time, parameter_server):
        self.name = name
        self.sess = sess
        self.game = game_instance
        self.game.game.set_seed(random_seed)
        self.game.game.set_render_weapon(True)
        self.game.game.set_render_crosshair(True)
        self.game.game.set_episode_timeout(500)
        self.game.game.init()
        self.network = network
        self.agent = agent
        
        self.clear_obs()
        self.clear_batch()
        
        self.start_time = start_time
        self.end_time = end_time
        self.n_step = n_step
        self.progress = 0.0
        self.log_server = None
        
        self.replay_memory = None
        
        self.step = 0
        self.model_gen_count = 0
        
        self.times_act = None
        self.times_update = None
        
        self.count_update = 0
        self.rewards_detail = None
        
        self.count_idx = np.zeros_like(replaymemory.tree.tree, dtype=np.int32)
        print(self.name," initialized...")
        
    def run_learning(self, coordinator):
        print(self.name + " start learning")
        self.network.pull_parameter_server(self.sess)
        self.network.copy_network_learning2target(self.sess)
        self.game.new_episode()
        try:
            while not coordinator.should_stop():
                self.learning_step()
                if self.n_step is not None:
                    self.progress = self.step/self.n_step
                else:
                    self.progress = (datetime.now().timestamp() - self.start_time)/(self.end_time - self.start_time)
#                 if self.progress >= 1.0:
#                     break
        except Exception as e:
            print(e)
            print(self.name," ended")
            
#         if self.log_server is not None:
#             coordinator.request_stop()

        return 0
    
    def run_prelearning(self, coordinator):
        assert self.replay_memory is not None
        self.network.pull_parameter_server(self.sess)
        self.network.copy_network_learning2target(self.sess)
        try:
            while not coordinator.should_stop():
                loss_values = self.prelearning_step()
                if self.n_step is not None:
                    self.progress = self.step/self.n_step
                else:
                    self.progress = (datetime.now().timestamp() - self.start_time)/(self.end_time - self.start_time)
        except Exception as e:
            coordinator.request_stop(e)
            
        coordinator.request_stop()
        return 0
    
    def run_exploring(self, coordinator):
        print(self.name + " start exploring")
        self.network.pull_parameter_server(self.sess)
        self.network.copy_network_learning2target(self.sess)
        self.game.new_episode()
        try:
            while not coordinator.should_stop():
                self.exploring_step()
                if self.n_step is not None:
                    if self.step % 1000 == 0:
                        print(self.name,":", self.step)
                    self.progress = self.step/self.n_step
                else:
                    self.progress = (datetime.now().timestamp() - self.start_time)/(self.end_time - self.start_time)
                if self.progress >= 1.0:
                    break
        except Exception as e:
            coordinator.request_stop(e)
            
        if self.log_server is not None:
            coordinator.request_stop()

        return 0
    
    def run_test(self, coordinator):
        self.network.pull_parameter_server(self.sess)
        self.network.copy_network_learning2target(self.sess)
        try:
            while not coordinator.should_stop():
#             while True:
                play_log = []
                reward,frag, death,kill,total_detail,step = self.test_agent(reward_buff =play_log)
                with open(os.path.join(PLAY_LOGDIR, "playlog_step%02d.txt"%int(self.progress*100)), 'wb') as f:
                    pickle.dump(play_log, f)
                if self.rewards_detail is not None:
                    self.rewards_detail.append(total_detail)
#                 print("----------TEST at %.1f ---------"%(self.progress*100))
#                 print("FRAG:",frag,"KILL:",kill, "DEATH:",death,"STEP:",step)
#                 print("REWARD:",reward)
#                 print("REWARD_DETAIL", total_detail)

                if self.log_server is not None:
                    self.log_server.write_score(self.sess,self.step,  reward, frag, death ,kill, step)
                    if self.progress >= self.model_gen_count/12:
                        self.model_gen_count += 1
                        self.log_server.save_model(sess=self.sess, model_path=MODEL_PATH, step=self.model_gen_count+1)
                        
                    
                self.step += 1
                if self.n_step is not None:
                    self.progress = self.step/self.n_step
                else:
                    self.progress = (datetime.now().timestamp() - self.start_time)/(self.end_time - self.start_time)
                if self.progress >= 1.0:
                    break
        except Exception as e:
            print(self.name, "killed ")
#             coordinator.request_stop(e)

    def learning_step(self):
        self.network.pull_parameter_server(self.sess)
#         self.network.push_parameter_server(self.sess)
        loss_values = []
        if not self.game.is_episode_finished() and self.game.get_screen_buff() is not None:
            
            if self.times_act is not None:
                start_time = datetime.now().timestamp()

            s1_ = self.preprocess(self.game.get_screen_buff())
            self.push_obs(s1_)
            agent_action_idx = self.agent.act_eps_greedy(self.sess, self.obs['s1'], self.progress)
#             engin_action = self.convert_action_agent2engine_simple(agent_action_idx)
            engin_action = self.convert_action_agent2engine(agent_action_idx)
            r,r_detail = self.game.make_action(self.step,engin_action , FRAME_REPEAT)
            isterminal = self.game.is_episode_finished()
            if isterminal:
                s2_ = np.zeros(RESOLUTION)
            else:
                s2_ = self.preprocess(self.game.get_screen_buff())
            
            self.push_batch( self.obs['s1'], agent_action_idx, s2_, r , isterminal, False)
            
            if self.times_act is not None:
                self.times_act.append(datetime.now().timestamp() - start_time)
            
            if len(self.memory) >= N_ADV or isterminal:
                batch = self.make_advantage_data()
                self.clear_batch()
                for i,b in enumerate(batch):
                    if len(b) == 8:
                        self.replay_memory.store(b)
            
            self.step += 1
            
            if self.step % INTERVAL_UPDATE_NETWORK == 0:
                self.network.copy_network_learning2target(self.sess)
                
            if self.times_update is not None:
                start_time = datetime.now().timestamp()
            
            if self.step % INTERVAL_BATCH_LEARNING == 0 and len(self.replay_memory) >= N_BATCH:
                s1, actions, r_one, r_adv, isdemo, is_weight, tree_idx = self.make_batch()
                if self.log_server is not None:
                    self.count_idx[tree_idx] += 1
                loss_values = self.network.update_parameter_server(self.sess, s1, actions, r_one, r_adv, isdemo, is_weight)
                self.count_update += 1
                tderror = loss_values[4]
                l_one, l_n, l_m, l_l = loss_values[:-1]
#                 self.replay_memory.batch_update(tree_idx, tderror)
#                 self.replay_memory.batch_update_new(tree_idx, tderror,np.array(r_adv)>0)
                if self.log_server is not None:
                    self.log_server.write_loss(self.sess,self.step ,np.mean(l_one), np.mean(l_n), np.mean(l_m), l_l)
#                     self.log_server.write_img(self.sess, self.step, s1[0:1])
                    self.log_server.write_weights(self.sess, self.step)
                self.replay_memory.batch_update_new(tree_idx, np.copy(l_one),np.array(r_adv)>0)
                    
            if self.times_update is not None:
                self.times_update.append(datetime.now().timestamp() - start_time)
        else:
            self.game.new_episode()
            self.clear_batch()
            self.clear_obs()

        return loss_values
    
    def prelearning_step(self):
        self.network.pull_parameter_server(self.sess)
#         self.network.push_parameter_server(self.sess)

        s1, actions, r_one, r_adv, isdemo, is_weight, tree_idx = self.make_batch()
        loss_values = self.network.update_parameter_server(self.sess, s1, actions, r_one, r_adv, isdemo, is_weight)
        tderror = loss_values[4]
        l_one, l_n, l_m, l_l = loss_values[:-1]
        self.replay_memory.batch_update(tree_idx, tderror) 
        
        if self.step % INTERVAL_UPDATE_NETWORK == 0:
            self.network.copy_network_learning2target(self.sess)
        
        if self.log_server is not None:
            if self.step % 10 == 0:
                self.log_server.write_loss(self.sess, self.step, np.mean(l_one), np.mean(l_n), np.mean(l_m), l_l)
#                 self.log_server.write_img(self.sess, self.step, s1[0:1])
                self.log_server.write_weights(self.sess, self.step)
        self.step += 1
        return loss_values

    def test_agent(self, gif_buff=None, reward_buff=None, sample_imgs=None):
        
        self.game.new_episode()
        self.network.pull_parameter_server(self.sess)

        step = 0
        gif_img = []
        total_reward = 0
        total_detail = {}
        self.clear_obs()
        while not self.game.is_episode_finished():
            s1_row = self.game.get_screen_buff()
            s1 = self.preprocess(s1_row)
            if sample_imgs is not None:
                sample_imgs.append(s1)
            if gif_buff is not None:
                gif_img.append(s1_row.transpose(1,2,0))
            self.push_obs(s1)
            action = self.agent.act_greedy(self.sess,self.obs['s1'])
#             engine_action = self.convert_action_agent2engine_simple(action)
            engine_action = self.convert_action_agent2engine(action)
            reward,reward_detail = self.game.make_action(step,engine_action,FRAME_REPEAT)
            isterminal = self.game.is_episode_finished()
            total_reward += reward
            for k in reward_detail.keys():
                if not k in total_detail.keys():
                    total_detail[k] = reward_detail[k]
                else:
                    total_detail[k] += reward_detail[k]
            step += 1
            if reward_buff is not None:
                reward_buff.append((engine_action, reward_detail))
        
        save_img = []
        if gif_buff is not None:
            for i in range(len(gif_img)):
                save_img.append(Image.fromarray(np.uint8(gif_img[i])))
            gif_buff += save_img
            
        return total_reward, self.game.get_frag_count(), self.game.get_death_count(), self.game.get_kill_count(), total_detail, step
        
    def convert_action_engine2agent(self,engine_action):
#         return engine_action.index(1)
        assert type(engine_action) == type(list()), print("type: ", type(engine_action))
        ans = 0
        for i, e_a in enumerate(engine_action):
            ans += e_a * 2**i
        return ans
    
    def convert_action_agent2engine(self,agent_action):
        assert type(agent_action) == type(int()) or type(agent_action) == type(np.int64()), print("type(agent_action)=",type(agent_action))
        ans = []
        for i in range(N_ACTION):
            ans.append(agent_action%2)
            agent_action = int(agent_action / 2)
        return ans
    
    def convert_action_agent2engine_simple(self, agent_action):
        assert type(agent_action) == type(int()) or type(agent_action) == type(np.int64()), print("type(agent_action)=",type(agent_action))
        ans = np.zeros((N_AGENT_ACTION,))
        ans[agent_action] = 1
        return ans.tolist()
    
    def preprocess(self,img):
        if len(img.shape) == 3 and img.shape[0]==3:
            img = img.transpose(1,2,0)

        img = skimage.transform.resize(img, RESOLUTION, mode="constant")
        img = img.astype(np.float32)
#         img = (img)/255.0
        return img

    def push_obs(self, s1):
        self.obs['s1'] = s1
        
    def clear_obs(self):
        self.obs = {}
        self.obs['s1'] = np.zeros(RESOLUTION, dtype=np.float32)
        
    def push_batch(self, s1, action,s2,  reward, isterminal, isdemo):
        self.memory.append([np.copy(s1), action, np.copy(s2) , reward, isterminal, isdemo])
    
    def clear_batch(self):
        self.memory = []
    
    def make_advantage_data(self):
        len_memory = len(self.memory)
        ret_batch = []
        R_adv = 0
        _,_,s2_adv,_,_,_ = self.memory[-1]
        for i in range(len_memory-1, -1, -1):
            s1,a,s2,r,isterminal,isdemo = self.memory[i]
            R_adv = r + GAMMA*R_adv
            ret_batch.append(np.array([s1, a,s2,s2_adv,r ,R_adv ,isterminal, isdemo]))
        
        self.memory = []
        return ret_batch
    
    def make_batch(self):
        while True:
            tree_idx, batch_row, is_weight = self.replay_memory.sample(N_BATCH, self.calc_beta(self.progress))
#             tree_idx, batch_row, is_weight = self.replay_memory.sample(N_BATCH, 0.1)
            s2_input = [ batch_row[i,2] for i in range(N_BATCH)]
            s2_adv = [ batch_row[i,3] for i in range(N_BATCH)]
            if (np.shape(s2_input) == ((N_BATCH,)+RESOLUTION) and np.shape(s2_adv) == ((N_BATCH,)+RESOLUTION)):
                break
        
        s1, actions, s2, r_one, r_adv, isdemo = [],[],[],[],[],[]
        
        predicted_q_adv  = self.network.get_qvalue_max_learningaction(self.sess,s2_adv)
        
        predicted_q = self.network.get_qvalue_max_learningaction(self.sess,s2_input)
        
        for i in range(N_BATCH):
            s1.append(batch_row[i][0])
            actions.append(batch_row[i][1])
            R_one = batch_row[i][4] + GAMMA * predicted_q[i] if batch_row[i][6] == False else batch_row[i][4]
            R_adv = batch_row[i][5] + GAMMA**N_ADV * predicted_q_adv[i] if batch_row[i][6] == False else batch_row[i][5]
            r_one.append(R_one)
            r_adv.append(R_adv)
            isdemo.append(batch_row[i][7])

        actions = np.array(actions)
        return s1, actions.astype(np.int32), r_one, r_adv, isdemo, is_weight, tree_idx
    
    def make_batch_uniform(self):
        while True:
            tree_idx, batch_row, is_weight = self.replay_memory.sample_uniform(N_BATCH)
            
            s2_input = [ batch_row[i,2] for i in range(N_BATCH)]
            s2_adv = [ batch_row[i,3] for i in range(N_BATCH)]
            if (np.shape(s2_input) == (N_BATCH,5, 120,120,3) and np.shape(s2_adv) == (N_BATCH,5, 120,120,3)):
                break
        
        s1, actions, s2, r_one, r_adv, isdemo = [],[],[],[],[],[]
        
        predicted_q_adv  = self.network.get_qvalue_max_learningaction(self.sess,s2_adv)
        
        predicted_q = self.network.get_qvalue_max_learningaction(self.sess,s2_input)
        
        for i in range(N_BATCH):
            s1.append(batch_row[i][0])
            actions.append(batch_row[i][1])
            R_one = batch_row[i][4] + GAMMA * predicted_q[i] if batch_row[i][6] == False else batch_row[i][4]
            R_adv = batch_row[i][5] + GAMMA**N_ADV * predicted_q_adv[i] if batch_row[i][6] == False else batch_row[i][5]
            r_one.append(R_one)
            r_adv.append(R_adv)
            isdemo.append(batch_row[i][7])

        actions = np.array(actions)
        return s1, actions.astype(np.int32), r_one, r_adv, isdemo, is_weight, tree_idx
    
    def calc_beta(self, progress):
#         return BETA_MIN
        return (BETA_MAX - BETA_MIN) * progress + BETA_MIN
    
    def exploring_step(self):
        if self.step % INTERVAL_PULL_PARAMS == 0:
            self.network.pull_parameter_server(self.sess)
        loss_values = []
        if not self.game.is_episode_finished() and self.game.get_screen_buff() is not None:

            s1_ = self.preprocess(self.game.get_screen_buff())
            self.push_obs(s1_)
            agent_action_idx = self.agent.act_eps_greedy(self.sess, self.obs['s1'], self.progress)
#             engin_action = self.convert_action_agent2engine_simple(agent_action_idx)
            engin_action = self.convert_action_agent2engine(agent_action_idx)
            r,r_detail = self.game.make_action(self.step,engin_action , FRAME_REPEAT)
            isterminal = self.game.is_episode_finished()
            if isterminal:
                s2_ = np.zeros(RESOLUTION)
            else:
                s2_ = self.preprocess(self.game.get_screen_buff())
            
            self.push_batch( self.obs['s1'], agent_action_idx, s2_, r , isterminal, False)
            
            if len(self.memory) >= N_ADV or isterminal:
                batch = self.make_advantage_data()
                self.clear_batch()
                for i,b in enumerate(batch):
                    if len(b) == 8:
                        self.replay_memory.store(b)
            
            self.step += 1
        else:
            self.game.new_episode()
            self.clear_batch()
            self.clear_obs()

        return loss_values
        


# In[ ]:


class ParameterServer:
    def __init__(self, sess, log_dir):
        self.sess = sess
        with tf.variable_scope("parameter_server", reuse=tf.AUTO_REUSE):
            self.state1_ = tf.placeholder(tf.float32, shape=(None,) + RESOLUTION)
            self.q_value, self.conv1, self.conv2, self.q_prob = self._build_model(self.state1_)

        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="parameter_server")
#         self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, RMSProbDecaly)
        self.optimizer = tf.train.AdamOptimizer()
            
        with tf.variable_scope("summary", reuse=tf.AUTO_REUSE):
            self._build_summary(sess,log_dir)
        
        self.saver = tf.train.Saver(max_to_keep = 20)
        
#         print("-------GLOBAL-------")
#         for w in self.weights_params:
#             print(w)

    def _build_model(self,state):
            conv1 = NetworkSetting.conv1(state)
#             maxpool1 = NetworkSetting.maxpool1(conv1)
            conv2 = NetworkSetting.conv2(conv1)
#             maxpool2 = NetworkSetting.maxpool2(conv2)
            reshape = NetworkSetting.reshape(conv2)
            fc1 = NetworkSetting.fc1(reshape)
            q = NetworkSetting.q_value(fc1)
            
            q_prob = tf.nn.softmax(q)
                
            print("---------MODEL SHAPE-------------")
            print(state.get_shape())
            print(conv1.get_shape())
#             print(maxpool1.get_shape())
            print(conv2.get_shape())
#             print(maxpool2.get_shape())
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
        self.loss_one_ = tf.placeholder(tf.float32, shape=(), name="loss_one")
        self.loss_adv_ = tf.placeholder(tf.float32, shape=(), name="loss_adv")
        self.loss_cls_ = tf.placeholder(tf.float32, shape=(), name="loss_class")
        self.loss_l2_ = tf.placeholder(tf.float32, shape=(), name="loss_l2")
        
        with tf.variable_scope("Summary_Score"):
            s = [tf.summary.scalar('reward', self.reward_, family="score"), tf.summary.scalar('frag', self.frag_, family="score"),                  tf.summary.scalar("death", self.death_, family="score"), tf.summary.scalar("kill", self.kill_, family="score"),                  tf.summary.scalar("step",self.score_step_, family="score")]
            self.summary_reward = tf.summary.merge(s)
        
        with tf.variable_scope("Summary_Loss"):
            list_summary = [tf.summary.scalar('loss_onestep', self.loss_one_, family="loss"), tf.summary.scalar('loss_advantage', self.loss_adv_, family="loss"), tf.summary.scalar('loss_class', self.loss_cls_, family="loss"), tf.summary.scalar('loss_l2', self.loss_l2_, family='loss')]
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
        
    def write_score(self,sess, step ,reward, frag, death, kill, score_step):
        m = sess.run(self.summary_reward, feed_dict={self.reward_:reward, self.frag_:frag, self.death_:death, self.kill_:kill, self.score_step_:score_step})
        return self.writer.add_summary(m, step)
    
    def write_loss(self,sess, step, l_o, l_n,l_c, l_l):
        m = sess.run(self.summary_loss, feed_dict={self.loss_one_: l_o, self.loss_adv_:l_n, self.loss_cls_:l_c, self.loss_l2_:l_l})
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


# In[ ]:


class Agent(object):
    
    def __init__(self, network,random_seed):
        self.network = network
        self.randomstate = np.random.RandomState(random_seed)
        
    def calc_eps(self, progress):
        if progress < 0.2:
            return EPS_MIN
        elif progress >= 0.2 and progress < 0.8:
            return ((EPS_MAX - EPS_MIN)/ 0.6) * progress + ( EPS_MIN -  (EPS_MAX - EPS_MIN)/ 0.6 * 0.2)
        else :
            return EPS_MAX

    def act_eps_greedy(self, sess, s1, progress):
        assert progress >= 0.0 and progress <=1.0
        
        eps = self.calc_eps(progress)
        if self.randomstate.rand() <= eps:
            a_idx = self.randomstate.choice(range(N_AGENT_ACTION), p=self.network.get_policy(sess,[s1])[0])
#             a_idx = self.network.get_best_action(sess, [s1])[0]
        else:
            a_idx = self.randomstate.randint(N_AGENT_ACTION)
            
        return a_idx
    
    def act_greedy(self, sess, s1):
        a_idx = self.randomstate.choice(range(N_AGENT_ACTION), p=self.network.get_policy(sess,[s1])[0])
#         a_idx = self.network.get_best_action(sess, [s1])[0]
        return a_idx
    
    def get_sum_prob(self,sess, s1):
        q_value = self.network.get_qvalue_learning(sess, [s1])[0]
        q_value = np.maximum(q_value,0) + 0.01
        q_prob = (q_value)/sum(q_value)
        a_idx = np.random.choice(N_AGENT_ACTION, p=q_prob)
        return a_idx


# In[ ]:


class NetworkLocal(object):
    def __init__(self,name, parameter_server):
        self.name = name
        
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("learning_network"):
                self.state1_ = tf.placeholder(tf.float32,shape=(None,)+RESOLUTION, name="state_1")
                self.q_value, self.conv1, self.conv2,self.reshape,self.fc1 = self._build_model(self.state1_)
            with tf.variable_scope("target_network"):
                self.state1_target_ = tf.placeholder(tf.float32,shape=(None,)+RESOLUTION, name="state_1")
                self.q_value_target,_,_,_,_ = self._build_model(self.state1_target_)
            
            self.a_ = tf.placeholder(tf.int32, shape=(None,), name="action")
            self.target_one_ = tf.placeholder(tf.float32, shape=(None,), name="target_one_")
            self.target_n_ = tf.placeholder(tf.float32, shape=(None,), name="target_n_")
            self.isdemo_ = tf.placeholder(tf.float32,shape=(None,), name="isdemo_")
            self.mergin_ = tf.placeholder(tf.float32,shape=(None,N_AGENT_ACTION), name="mergin_")
            self.is_weight_ = tf.placeholder(tf.float32, shape=(None,), name="is_weight")
        
                
            self._build_graph()
            
#             self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, RMSProbDecaly)
#             self.optimizer = tf.train.AdamOptimizer()
#             self.update = self.optimizer.minimize(self.loss_total, var_list = self.weights_params_learning)
        
#         self.grads = parameter_server.optimizer.compute_gradients(self.loss_total, var_list=self.weights_params_learning)
        
        self.update_global_weight_params =             parameter_server.optimizer.apply_gradients([(g,w) for g, w in zip(self.grads, parameter_server.weights_params)])
        
        self.pull_global_weight_params = [l_p.assign(g_p) for l_p,g_p in zip(self.weights_params_learning,parameter_server.weights_params)]
        self.push_local_weight_params = [g_p.assign(l_p) for g_p,l_p in zip(parameter_server.weights_params,self.weights_params_learning)]

    def _build_model(self,state):
        conv1 = NetworkSetting.conv1(state)
#         maxpool1 = NetworkSetting.maxpool1(conv1)
        conv2 = NetworkSetting.conv2(conv1)
#         maxpool2 = NetworkSetting.maxpool2(conv2)
        reshape = NetworkSetting.reshape(conv2)
        fc1 = NetworkSetting.fc1(reshape)
        
        q_value = NetworkSetting.q_value(fc1)
        
        return q_value, conv1, conv2,reshape,fc1

    def _build_graph(self):

        self.q_prob = tf.nn.softmax(self.q_value)
        self.q_argmax = tf.argmax(self.q_value, axis=1)
        self.q_learning_max = tf.reduce_max(self.q_value, axis=1)
        self.q_target_max = tf.reduce_max(self.q_value_target, axis=1)
        
        self.weights_params_learning = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+"/learning_network")
        self.weights_params_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+"/target_network")
        
        self.tderror_one = LAMBDA_ONE * tf.abs(self.target_one_ - tf.reduce_max(self.q_value, axis=1))
        self.loss_one = (LAMBDA_ONE * tf.square(self.target_one_ - tf.reduce_max(self.q_value, axis=1))) * self.is_weight_
        self.tderror_n = LAMBDA1 * tf.abs(self.target_n_ - tf.reduce_max(self.q_value, axis=1))
        self.loss_n = (LAMBDA1 * tf.square(self.target_n_ - tf.reduce_max(self.q_value, axis=1)))*self.is_weight_
        self.loss_l2 = LAMBDA3 * tf.reduce_sum([tf.nn.l2_loss(w) for w in self.weights_params_learning])
        
        idx = tf.transpose([tf.range(tf.shape(self.q_value)[0]), self.a_])
        self.loss_mergin = LAMBDA2 * ((tf.stop_gradient(tf.reduce_max(self.q_value + self.mergin_, axis=1)) - tf.gather_nd(self.q_value,indices=idx))*self.isdemo_)
        
        self.tderror_total = self.tderror_one + self.tderror_n + self.loss_mergin
        self.loss_total = tf.reduce_mean(self.loss_one +  self.loss_n + self.loss_mergin + self.loss_l2)
#         self.tderror_total = self.tderror_n
#         self.loss_total = tf.reduce_mean(self.loss_n+ self.loss_mergin + self.loss_l2)
    
        
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
        mergin_value = np.ones((len(s1), N_AGENT_ACTION)) * MERGIN_VALUE
        mergin_value[range(len(s1)), a] = 0.0
        feed_dict = {self.state1_: s1,self.a_:a, self.target_one_:target_one, self.target_n_:target_n, self.isdemo_:isdemo, self.is_weight_:is_weight, self.mergin_:mergin_value}
#         l_one, l_n, l_mergin, l_l2, tderror_total = sess.run([self.loss_one, self.loss_n, self.loss_mergin, self.loss_l2, self.tderror_total], feed_dict)
        l_one, tderror_total = sess.run([self.loss_n, self.tderror_n], feed_dict)
        return l_one, 0,0,0, tderror_total
    
    def get_losstotal(self, sess,s1, a, target_one,target_n, isdemo, is_weight):
        mergin_value = np.ones((len(s1), N_AGENT_ACTION)) * MERGIN_VALUE
        mergin_value[range(len(s1)), a] = 0.0
        feed_dict = {self.state1_: s1,self.a_:a, self.target_one_:target_one, self.target_n_:target_n, self.isdemo_:isdemo, self.is_weight_:is_weight, self.mergin_:mergin_value}
        loss_total = sess.run([self.loss_total], feed_dict)
        return loss_total[0]
    
    def get_grads(self, sess,s1, a, target_one,target_n, isdemo, is_weight):
        mergin_value = np.ones((len(s1), N_AGENT_ACTION)) * MERGIN_VALUE
        mergin_value[range(len(s1)), a] = 0.0
        feed_dict = {self.state1_: s1,self.a_:a, self.target_one_:target_one, self.target_n_:target_n, self.isdemo_:isdemo, self.is_weight_:is_weight, self.mergin_:mergin_value}
        grads = sess.run(self.grads, feed_dict)
        return grads
    
    def update_parameter_server(self, sess, s1, a, target_one,target_n, isdemo, is_weight):
        assert np.ndim(s1) == 4
        mergin_value = np.ones((len(s1), N_AGENT_ACTION)) * MERGIN_VALUE
        mergin_value[range(len(s1)), a] = 0.0
        feed_dict = {self.state1_: s1,self.a_:a, self.target_one_:target_one, self.target_n_:target_n, self.isdemo_:isdemo, self.is_weight_:is_weight, self.mergin_:mergin_value}
#         _,l_one, l_n, l_mergin, l_l2, tderror_total = sess.run([self.update, self.loss_one, self.loss_n, self.loss_mergin, self.loss_l2, self.tderror_total], feed_dict)
#         _,l_one,l_mergin,l_l2 ,tderror_total = sess.run([self.update_global_weight_params,self.loss_n,self.loss_mergin,self.loss_l2, self.tderror_total], feed_dict)
#         return l_one, 0,l_mergin,l_l2, tderror_total
        _,l_one, l_n, l_mergin, l_l2, tderror_total = sess.run([self.update_global_weight_params, self.loss_one, self.loss_n, self.loss_mergin, self.loss_l2, self.tderror_total], feed_dict)
        return l_one, l_n, l_mergin, l_l2, tderror_total
    
    def check_weights(self, sess):
        weights = SESS.run(self.weights_params_learning)
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


# In[ ]:


class NetworkSetting:
    
    def conv1(pre_layer):
        num_outputs = 16
        kernel_size = [6,6]
        stride = [3,3]
        padding = 'SAME'
        activation = tf.nn.relu
        weights_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(0.1)
        return tf.layers.conv2d(pre_layer,kernel_size=kernel_size,                                        filters=num_outputs,                                        strides=stride,padding=padding,activation=activation,                                        kernel_initializer=weights_init,                                        bias_initializer=bias_init)
    
    def maxpool1(pre_layer):
        return tf.nn.max_pool2d(pre_layer,[1,3,3,1],[1,2,2,1],'SAME')
    
    def conv2(pre_layer):
        num_outputs = 16
        kernel_size = [3,3]
        stride = [2,2]
        padding = 'SAME'
        activation = tf.nn.relu
        weights_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(0.1)
        return tf.layers.conv2d(pre_layer,kernel_size=kernel_size,filters=num_outputs,                                        strides=stride,padding=padding,activation=activation,                                        kernel_initializer=weights_init,bias_initializer=bias_init)
    
    def maxpool2(pre_layer):
        return tf.nn.max_pool2d(pre_layer,[1,1,3,3,1],[1,1,2,2,1],'SAME')
        
    def reshape(pre_layer):
        shape = pre_layer.get_shape()
#         return tf.reshape(pre_layer, shape=(-1,shape[1], shape[2]*shape[3]*shape[4]))
        return tf.reshape(pre_layer, shape=(-1,shape[1]*shape[2]*shape[3]))

    
    def fc1(pre_layer):
        num_outputs =1024
        activation_fn = tf.nn.relu
        weights_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.1)
        return tf.contrib.layers.fully_connected(pre_layer,num_outputs=num_outputs,activation_fn=activation_fn,weights_initializer=weights_init, biases_initializer=bias_init)
    
    def q_value(pre_layer):
        num_outputs=N_AGENT_ACTION
        activation_fn = None
        weights_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.1)
        return tf.contrib.layers.fully_connected(pre_layer,num_outputs=num_outputs,activation_fn=activation_fn,weights_initializer=weights_init, biases_initializer=bias_init)


# In[ ]:


def load_demo_one(replay_memory, demo_path):
    for demo in demo_path[:]:
        print(demo)
        file = h5py.File(demo, 'r')
        episodes = list(file.keys())[1:]
        game = GameInstanceBasic(DoomGame(),name="noname",n_bots=1,config_path=CONFIG_FILE_PATH, reward_param=REWARDS, steps_update_origin=10,timelimit=2)
        total_n_transit = 0
        for e in episodes:
            states_row = file[e+"/states"][:]
            action_row = file[e+"/action"][:]
#             action_row = action_row[:,[2,4,6]]
            action_row = action_row[:,[0,1,2,3,4,6]]
            
            n_transit = len(states_row)
            total_n_transit += n_transit

            memory = []
            
            for i in range(0, n_transit):
                
                s1_ = states_row[i]
                if i == n_transit -1 :
                    isterminal = True
                    s2_ = np.zeros(RESOLUTION)
                    r = REWARDS['kill'] + REWARDS['enemysight']
                else:
                    isterminal = False
                    s2_ = states_row[i+1]
                    r = REWARDS['living'] + REWARDS['enemysight']
                
#                 action = np.where(action_row[i]==1)[0][0]
                action = 0
                for i, e_a in enumerate(action_row[i]):
                    action += e_a * 2**i
                
                memory.append([s1_,action, s2_, r, isterminal, True])
                
                if len(memory) == N_ADV or isterminal==True:
                    R_adv = 0
                    len_memory = len(memory)
                    _, _, s2_adv, _, _, _ = memory[-1]
                    for i in range(len_memory - 1, -1, -1):
                        s1,a, s2 ,r,isterminal,isdemo = memory[i]
                        R_adv = r + GAMMA*R_adv
                        replaymemory.store(np.array([s1, a,s2 ,s2_adv,r ,R_adv ,isterminal, isdemo]))
                    memory = []
            
        file.close()
    replay_memory.set_permanent_data(total_n_transit)
    print(len(replay_memory), "data are stored")
    file.close()
    return 0


# In[ ]:


def load_positivedata(replay_memory, data_path_list):
    for data_path in data_path_list:
        print(data_path)
        p_data = np.load(data_path)
        for d in p_data:
            replay_memory.store(d)
    n_data = len(replay_memory)
    replay_memory.set_permanent_data(n_data)
    print(n_data, "data are stored")
    return 0


# In[ ]:


if __name__=="learning_imitation":
    print(LOGDIR)
    replaymemory = ReplayMemory(10000)
    load_demo_one(replaymemory, DEMO_PATH)
    
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list=USED_GPU))
    config.log_device_placement = False
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    
    N_STEPS = 15000

    with tf.device('/gpu:0'):
        parameter_server = ParameterServer(sess,LOGDIR)

        starttime = datetime.now().timestamp()
        end_time = (datetime.now() + timedelta(minutes=30)).timestamp()
        
        coordinator = tf.train.Coordinator()

        name = "worker_imitation"
        game_instance = GameInstanceBasic(DoomGame(),name=name,n_bots=1,config_path=CONFIG_FILE_PATH, reward_param=REWARDS, steps_update_origin=10,timelimit=2)
        network = NetworkLocal(name, parameter_server)
        agent = Agent(network,random_seed=0)
#         imitation_env = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, start_time=starttime, end_time=end_time, random_seed=0)
        imitation_env = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, n_step=N_STEPS, random_seed=0)
        imitation_env.log_server = parameter_server
        imitation_env.replay_memory = replaymemory
        thread_imitation = threading.Thread(target=imitation_env.run_prelearning, args=(coordinator,))

        name = "test"
        game_instance = GameInstanceBasic(DoomGame(),name=name,n_bots=1,config_path=CONFIG_FILE_PATH, reward_param=REWARDS, steps_update_origin=10,timelimit=2)
        network = NetworkLocal(name, parameter_server)
        agent = Agent(network,random_seed=100)
        test_env = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, start_time=starttime, end_time=end_time, random_seed=100)
#         test_env = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, n_step=10000, random_seed=0)
        test_env.log_server = parameter_server
        thread_test = threading.Thread(target=test_env.run_test, args=(coordinator,))
        
        parameter_server.write_graph(sess)
        sess.run(tf.global_variables_initializer())
        

        print("-----Start IMITATION LEARNING----")
        threads = [thread_imitation,thread_test]
        for t in threads:
            t.start()
#         coordinator.join(threads)
        while True:
            time.sleep(10)
            if imitation_env.progress >= 1.0:
                coordinator.request_stop()
                break

        parameter_server.save_model(sess=sess, step=15, model_path=MODEL_PATH)

        print(LOGDIR)


# In[ ]:


if __name__=="learning_async":
    print(LOGDIR)
    replaymemory = ReplayMemory(50000)
    load_demo_one(replaymemory, DEMO_PATH)
#     load_positivedata(replaymemory, POSITIVEDATA_PATH)
    
    tf.set_random_seed(0)
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list=USED_GPU))
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    with tf.device('/gpu:0'):
        parameter_server = ParameterServer(sess,LOGDIR)

        starttime = datetime.now().timestamp()
        end_time = (datetime.now() + timedelta(minutes=10)).timestamp()
        
        coordinator = tf.train.Coordinator()
        
        environments, threads = [], []
        
        for i in range(N_WORKERS):
            name = "worker_%d"%(i+1)
            game_instance=GameInstanceBasic(DoomGame(),name=name,n_bots=1,config_path=CONFIG_FILE_PATH, reward_param=REWARDS, steps_update_origin=10,timelimit=2)
#             game_instance=GameInstanceSimpleBasic(DoomGame(),name=name,config_path=CONFIG_FILE_PATH)
            network = NetworkLocal(name, parameter_server)
            agent = Agent(network, random_seed=i)
#             e = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, start_time=starttime, end_time=end_time, random_seed=i)
            e = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, n_step=N_STEPS, random_seed=i)
            e.replay_memory = replaymemory
            environments.append(e)

        environments[0].log_server = parameter_server
        environments[0].times_act = []
        environments[0].times_update = []
        
#         name = "updating"
#         game_instance_update=GameInstanceBasic(DoomGame(),name=name,n_bots=1,config_path=CONFIG_FILE_PATH, reward_param=REWARDS, steps_update_origin=10,timelimit=2)
# #             game_instance=GameInstanceSimpleBasic(DoomGame(),name=name,config_path=CONFIG_FILE_PATH)
#         network_update = NetworkLocal(name, parameter_server)
#         agent_update = Agent(network, random_seed=99)
#         update_env = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, start_time=starttime, end_time=end_time, random_seed=99)
#         update_env.replay_memory = replaymemory
#         thread_update = threading.Thread(target=update_env.run_prelearning, args=(coordinator,))
        
#         update_env.log_server = parameter_server
#         threads.append(thread_update)

        name = "test"
#         test_seed = np.random.randint(1000)
        test_seed = 100
        game_instance=GameInstanceBasic(DoomGame(),name=name,n_bots=1,config_path=CONFIG_FILE_PATH, reward_param=REWARDS, steps_update_origin=10,timelimit=2)
#         game_instance=GameInstanceSimpleBasic(DoomGame(),name=name,config_path=CONFIG_FILE_PATH)
        network = NetworkLocal(name, parameter_server)
        agent = Agent(network, random_seed=test_seed)
#         test_env = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, start_time=starttime, end_time=end_time, random_seed=test_seed)
        test_env = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, n_step=N_STEPS, random_seed=test_seed)
        test_env.log_server = parameter_server
        test_env.rewards_detail = []
        thread_test = threading.Thread(target=test_env.run_test, args=(coordinator,))

    for e in environments:
#         threads.append(threading.Thread(target=e.run_exploring, args=(coordinator,)))
            threads.append(threading.Thread(target=e.run_learning, args=(coordinator,)))

    threads.append(thread_test)

    parameter_server.write_graph(sess)
    sess.run(tf.global_variables_initializer())


#         parameter_server.load_model(sess=sess, step=15, model_path="./models/model_imitation181221/model.ckpt")
#         parameter_server.load_model(sess=sess, step=15, model_path="./models/largebasic_random/model_largebasicrandom_imitation190109/model.ckpt")
#         parameter_server.load_model(sess=sess, step=15, model_path="models/model_temp/model_2019-01-16-15-32-54/model.ckpt")
    print(IMIT_MODEL_PATH)
    parameter_server.load_model(sess=sess, step=15, model_path=IMIT_MODEL_PATH)
#     parameter_server.load_model(sess=sess, step=15, model_path="./models/model_temp/model_2019-01-27-15-58-51/model.ckpt")
    

    print("-----Start ASYNC LEARNING----")
    for t in threads:
        t.start()
#     coordinator.join(threads)
    while True:
        time.sleep(10)
        if np.array([e.progress >= 1.0 for e in environments]).all():
            coordinator.request_stop()
            break
    
    parameter_server.save_model(sess=sess, step=15, model_path=MODEL_PATH)

    GIF_BUFF = []
    REWARD_BUFF = []
    r,f,d,imgs,_,step = test_env.test_agent(gif_buff=GIF_BUFF,reward_buff=REWARD_BUFF)
    GIF_BUFF[0].save('gifs/test.gif',save_all=True, append_images=GIF_BUFF[1:], optimize=False, duration=40*4, loop=0)

    print(LOGDIR)
    print(sum([e.step for e in environments]))


# In[ ]:


if __name__=="test":
    print(LOGDIR)
    replaymemory = ReplayMemory(50000)
    load_demo_one(replaymemory, DEMO_PATH)
#     load_positivedata(replaymemory, POSITIVEDATA_PATH)
    
    tf.set_random_seed(0)
    
#     N_STEPS = 25000
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list=USED_GPU))
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    with tf.device('/gpu:0'):
        parameter_server = ParameterServer(sess,LOGDIR)

        starttime = datetime.now().timestamp()
        end_time = (datetime.now() + timedelta(minutes=60)).timestamp()
        
        coordinator = tf.train.Coordinator()

        name = "test"
#         test_seed = np.random.randint(1000)
        test_seed = 100
        game_instance=GameInstanceBasic(DoomGame(),name=name,n_bots=1,config_path=CONFIG_FILE_PATH, reward_param=REWARDS, steps_update_origin=10,timelimit=2)
#         game_instance=GameInstanceSimpleBasic(DoomGame(),name=name,config_path=CONFIG_FILE_PATH)
        network = NetworkLocal(name, parameter_server)
        agent = Agent(network, random_seed=test_seed)
        test_env = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, start_time=starttime, end_time=end_time, random_seed=test_seed)
#         test_env = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, n_step=N_STEPS, random_seed=test_seed)
        test_env.log_server = parameter_server

#         parameter_server.load_model(sess=sess, step=15, model_path="./models/model_imitation181221/model.ckpt")
#         parameter_server.load_model(sess=sess, step=15, model_path="./models/largebasic_random/model_largebasicrandom_imitation190109/model.ckpt")
#         parameter_server.load_model(sess=sess, step=15, model_path="models/model_temp/model_2019-01-16-15-32-54/model.ckpt")
    parameter_server.load_model(sess=sess, step=15, model_path="../data/demo_dqn/models/model_2019-01-31-10-44-57/model.ckpt")



# In[ ]:


def plot_priority(replaymemory):
    size = len(replaymemory)
    lengh = replaymemory.tree.capacity
    start_idx = lengh - 1
    end_idx = start_idx + size
    priority = replaymemory.tree.tree[start_idx:end_idx]
    plt.plot(priority)
    
def save_gif10(env):
    GIF_BUFF_TOTAL = []
    for i in range(10):
        buff = []
        val = test_env.test_agent(gif_buff=buff)
        print("REWARD:",val[0])
        GIF_BUFF_TOTAL = GIF_BUFF_TOTAL + buff
    GIF_BUFF_TOTAL[0].save('gifs/test.gif',save_all=True, append_images=GIF_BUFF_TOTAL[1:], optimize=False, duration=40*4, loop=0)

def plot_conv(env,s1):
    conv = sess.run(env.network.conv1,{env.network.state1_:[s1]})[0]
    display_img = conv[-1]
    print(display_img.shape)
    fig,axes = plt.subplots(4,8,figsize=(20,15))
    display_img = display_img.transpose((2,0,1))
    for ax,img in zip(axes.ravel(), display_img):
        ax.imshow(img)
        
def plot_conv_onw(env,s1):
    conv = sess.run(env.network.conv1,{env.network.state1_:[s1]})[0]
    display_img = conv
    print(display_img.shape)
    fig,axes = plt.subplots(4,8,figsize=(20,15))
    display_img = display_img.transpose((2,0,1))
    for ax,img in zip(axes.ravel(), display_img):
        ax.imshow(img)

def plot_q_learning(env, s1):
    q_value = env.network.get_qvalue_learning(sess,s1)
#     q_value = env.network.get_qvalue_target(sess,s1)
    fig,axes = plt.subplots(10,figsize=(20,20))
    for ax,q  in zip(axes.ravel(), q_value):
        ax.bar(range(len(q)), q)
        
    return q_value

def plot_q_target(env, s1):
    q_value = env.network.get_qvalue_target(sess,s1)
    fig,axes = plt.subplots(10,figsize=(20,20))
    for ax,q  in zip(axes.ravel(), q_value):
        ax.bar(range(len(q)), q)
        
    return q_value

def plot_q_softmax(env, s1):
    q_value = env.network.get_policy(sess,s1)
    fig,axes = plt.subplots(10,figsize=(20,20))
    for ax,q  in zip(axes.ravel(), q_value):
        ax.bar(range(len(q)), q)
        
    return q_value

def plot_diff_qvalue(env, s1):
    q_l = env.network.get_qvalue_learning(sess,s1)
    q_t = env.network.get_qvalue_target(sess,s1)
    fig,axes = plt.subplots(10,figsize=(20,20))
    for ax,q  in zip(axes.ravel(),q_t-q_l):
        ax.bar(range(len(q)), q)

def plot_s1(s1):
    fig,axes = plt.subplots(10,5,figsize=(20,20))
    for ax,s  in zip(axes,s1):
        for a,img in zip(ax,s):
            a.imshow(img)
            
def plot_s1_one(s1):
    fig,axes = plt.subplots(len(s1),figsize=(20,20))
    for ax,s  in zip(axes,s1):
        ax.imshow(s)

def plot_tderror(env, s1,action, s2_one,s2_adv,r_one, r_adv,isdemo, isterminal):
    predicted_q_adv  = env.network.get_qvalue_max_learningaction(sess,s2_adv)
    predicted_q_one = env.network.get_qvalue_max_learningaction(sess,s2_one)
    
    isnotterminal = np.ones((len(isterminal),)) - isterminal
    target_one = r_one + GAMMA*predicted_q_one * isnotterminal
    target_adv = r_adv + GAMMA**N_ADV * predicted_q_adv * isnotterminal
    action = [int(a) for a in action]
    isweight = np.ones((len(action),))
    loss_values = env.network.get_loss(sess, s1,action,target_one, target_adv,isdemo,isweight)
    plt.bar(range(len(isweight)),loss_values[-1])
    return loss_values[-1]

def plot_loss_one(env, s1,action, s2_one,s2_adv,r_one, r_adv,isdemo, isterminal):
    predicted_q_adv  = env.network.get_qvalue_max_learningaction(sess,s2_adv)
    predicted_q_one = env.network.get_qvalue_max_learningaction(sess,s2_one)
    
    isnotterminal = np.ones((len(isterminal),)) - isterminal
    target_one = r_one + GAMMA*predicted_q_one * isnotterminal
    target_adv = r_adv + GAMMA**N_ADV * predicted_q_adv * isnotterminal
    action = [int(a) for a in action]
    isweight = np.ones((len(action),))
    loss_values = env.network.get_loss(sess, s1,action,target_one, target_adv,isdemo,isweight)
    print(loss_values[0])
    print(np.mean(loss_values[0]))
    plt.bar(range(len(action)), loss_values[0])

def plot_loss_adv(env, s1,action, s2_one,s2_adv,r_one, r_adv,isdemo, isterminal):
    predicted_q_adv  = env.network.get_qvalue_max_learningaction(sess,s2_adv)
    predicted_q_one = env.network.get_qvalue_max_learningaction(sess,s2_one)
    
    isnotterminal = np.ones((len(isterminal),)) - isterminal
    target_one = r_one + GAMMA*predicted_q_one * isnotterminal
    target_adv = r_adv + GAMMA**N_ADV * predicted_q_adv * isnotterminal
    action = [int(a) for a in action]
    isweight = np.ones((len(action),))
    loss_values = env.network.get_loss(sess, s1,action,target_one, target_adv,isdemo,isweight)
    print(loss_values[1])
    print(np.mean(loss_values[1]))
    plt.bar(range(len(action)), loss_values[1])
    
def plot_losstotal(env, s1,action, s2_one,s2_adv,r_one, r_adv,isdemo, isterminal):
    predicted_q_adv  = env.network.get_qvalue_max_learningaction(sess,s2_adv)
    predicted_q_one = env.network.get_qvalue_max_learningaction(sess,s2_one)
    
    isnotterminal = np.ones((len(isterminal),)) - isterminal
    target_one = r_one + GAMMA*predicted_q_one * isnotterminal
    target_adv = r_adv + GAMMA**N_ADV * predicted_q_adv * isnotterminal
    action = [int(a) for a in action]
    isweight = np.ones((len(action),))
    loss_total = env.network.get_losstotal(sess, s1,action,target_one, target_adv,isdemo,isweight)
    print(loss_total)

def plot_loss_class(env, s1,action, s2_one,s2_adv,r_one, r_adv,isdemo, isterminal):
    predicted_q_adv  = env.network.get_qvalue_max_learningaction(sess,s2_adv)
    predicted_q_one = env.network.get_qvalue_max_learningaction(sess,s2_one)
    
    isnotterminal = np.ones((len(isterminal),)) - isterminal
    target_one = r_one + GAMMA*predicted_q_one * isnotterminal
    target_adv = r_adv + GAMMA**N_ADV * predicted_q_adv * isnotterminal
    action = [int(a) for a in action]
    isweight = np.ones((len(action),))
    loss_values = env.network.get_loss(sess, s1,action,target_one, target_adv,isdemo,isweight)
    print(loss_values[2])
    plt.bar(range(len(action)), loss_values[2])
    
def plot_freq_sample(replaymemory):
    idx = []
    demo_r = []
    for i in range(10000):
        tree_idx,data,_ = replaymemory.sample(1, 0.5)
        idx.append(tree_idx)
    idx = np.array(idx).reshape((-1))
    print(idx)
    count = np.zeros((len(replaymemory),))
    for i in idx:
        count[i-replaymemory.tree.capacity]+= 1
    plt.plot(count)
    
def play_games(env, t=50):
    kill,reward,step = [],[],[]
    for i in range(t):
        r,f, d,k,t,s = env.test_agent()
        kill.append(k)
        reward.append(r)
        step.append(s)
    return np.array(kill), np.array(reward), np.array(step)

def plot_filter_conv1(env):
    weights = env.network.get_weights_learngin(sess)
    weights_conv1 = weights[0]
    fig,axes = plt.subplots(3,16,figsize=(20,20))
    weights_conv1 = weights_conv1.transpose(2,3,0,1)
    for ax,img in zip(axes.ravel(), weights_conv1.reshape(-1,6,6)):
        ax.imshow(img)


# In[ ]:


# save_gif10(test_env)


# In[ ]:


# data_range = range(0,1000)
# # data_range = range(6000,6010)
# demo_s,demo_r_one,demo_r_adv, demo_s2_one, demo_s2_adv, demo_a,demo_t,demo_d = [],[],[],[],[],[],[],[]
# for i in data_range:
#     demo_s.append(replaymemory.tree.data[i][0])
#     demo_s2_one.append(replaymemory.tree.data[i][2])
#     demo_s2_adv.append(replaymemory.tree.data[i][3])
#     demo_a.append(replaymemory.tree.data[i][1])
#     demo_r_one.append(replaymemory.tree.data[i][4])
#     demo_r_adv.append(replaymemory.tree.data[i][5])
#     demo_t.append(replaymemory.tree.data[i][6])
#     demo_d.append(replaymemory.tree.data[i][7])


# In[ ]:


# plot_losstotal(env=imitation_env, s1=demo_s,action=demo_a,s2_one=demo_s2_one, s2_adv=demo_s2_adv, \
#              r_one=demo_r_one, r_adv=demo_r_adv ,isterminal=demo_t,isdemo=demo_d)

# for i,d in enumerate(zip(demo_a, demo_r_one)):
#     print(i,d[0],d[1])

# plot_tderror(env=environments[0], s1=demo_s,action=demo_a,s2_one=demo_s2_one, s2_adv=demo_s2_adv, \
#              r_one=demo_r_one, r_adv=demo_r_adv ,isterminal=demo_t,isdemo=demo_d)

# plot_loss_one(env=imitation_env, s1=demo_s,action=demo_a,s2_one=demo_s2_one, s2_adv=demo_s2_adv, \
#              r_one=demo_r_one, r_adv=demo_r_adv ,isterminal=demo_t,isdemo=demo_d)

# hoge = plot_q_learning(env=environments[0], s1=demo_s)

# kills, rewards,steps = play_games(test_env,t=100)

# steps[rewards < 0] = 100

# print(np.mean(kills),"+-",np.var(kills))
# print(np.mean(rewards),"+-",np.var(rewards))
# print(np.mean(steps),"+-",np.var(steps))


# In[ ]:





# tf.set_random_seed(0)
    
# config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list=USED_GPU))
# config.gpu_options.allow_growth = True
# config.log_device_placement = False
# config.allow_soft_placement = True
# sess = tf.Session(config=config)
# parameter_server = ParameterServer(sess,LOGDIR)

# starttime = datetime.now().timestamp()
# end_time = (datetime.now() + timedelta(minutes=60)).timestamp()

# coordinator = tf.train.Coordinator()

# environments, threads = [], []
# name = "test"
# game_instance=GameInstanceBasic(DoomGame(),name=name,n_bots=1,config_path=CONFIG_FILE_PATH, reward_param=REWARDS, steps_update_origin=10,timelimit=2)
# #         game_instance=GameInstanceSimpleBasic(DoomGame(),name=name,config_path=CONFIG_FILE_PATH)
# network = NetworkLocal(name, parameter_server)
# agent = Agent(network)
# test_env = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, start_time=starttime, end_time=end_time, random_seed=0)
# test_env.log_server = parameter_server
# test_env.rewards_detail = []
# thread_test = threading.Thread(target=test_env.run_test, args=(coordinator,))


# In[ ]:


if __name__ == "test":
    MODEL_PATH = "./models/model_test/model.ckpt"
    TEST_GRADS = []
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list=USED_GPU))
    config.log_device_placement = False
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    starttime = datetime.now().timestamp()
    end_time = (datetime.now() + timedelta(minutes=15)).timestamp()
    
    with tf.device('/gpu:0'):
        parameter_server = ParameterServer(sess,LOGDIR)
        parameter_server.load_model(sess=sess, model_path=MODEL_PATH, step=15)
        
        name = "test"
        game_instance = GameInstanceSimpleDeathmatch(DoomGame(),name=name,n_bots=1,config_path=CONFIG_FILE_PATH, reward_param=REWARDS, steps_update_origin=10,timelimit=2)
        network = NetworkLocal(name, parameter_server)
        agent = Agent(network)
        test_env = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, start_time=starttime, end_time=end_time)
        GIF_BUFF = []
        REWARD_BUFF = []
        r,f,d,_,imgs = test_env.test_agent(gif_buff=GIF_BUFF,reward_buff=REWARD_BUFF)
        GIF_BUFF[0].save('gifs/test.gif',save_all=True, append_images=GIF_BUFF[1:], optimize=False, duration=40*4, loop=0)


# In[ ]:


if __name__ == "test_game_instance":
    environments[0].game.new_episode()
    pre_x = 0
    pre_y = 0

    print(environments[0].game.get_pos_x(),"diff:",environments[0].game.get_pos_x()-pre_x, ",", environments[0].game.get_pos_y(),"diff:",environments[0].game.get_pos_y()-pre_y)
    pre_x = environments[0].game.get_pos_x()
    pre_y = environments[0].game.get_pos_y()
    print(environments[0].game.make_action([0,0,0,1,0,1], FRAME_REPEAT))
    # print(environments[0].game.game)
    plt.imshow(environments[0].preprocess( environments[0].game.get_screen_buff()))

    if(environments[0].game.is_player_dead()):
        environments[0].game.respawn_player()


# In[ ]:


if __name__ == "check_rewards":
    files = os.listdir("./playlogs/playlog_test/")
    files.sort()
    rewards = []
    for f in files[:]:
        with open(os.path.join("./playlogs/playlog_test/", f), 'rb') as file:
            rewards.append((f, pickle.load(file)))

def make_data(rewards_all):
    data_frames = []
    for f_n, reward in rewards_all:
        reward_dist = []
        reward_frag = []
        reward_healthloss = []
        reward_suicide = []
        reward_total = []
        for log in reward:
            _,r = log
            reward_dist.append(r['dist'])
            reward_frag.append(r['frag'])
            reward_healthloss.append(r['healthloss'])
            reward_suicide.append(r['suicide'])
            reward_total.append(sum(r.values()))
            
        df = pd.DataFrame({'dist':reward_dist, 'frag': reward_frag, 'healthloss':reward_healthloss, 'suicide':reward_suicide})
        data_frames.append(df)
    return data_frames

def plot_rewards_all(rewards_all):
    
    dist_all = [sum(r['dist'].values) for r in rewards_all]
    frag_all = [sum(r['frag'].values) for r in rewards_all]
    healthloss_all = np.array( [sum(r['healthloss'].values) for r in rewards_all])
    healthloss_all= np.where(healthloss_all < -100, 0, healthloss_all)
    suicide_all = [sum(r['suicide'].values) for r in rewards_all]
    total = [sum(r) for r in zip(dist_all, frag_all, healthloss_all, suicide_all)]
    
    f = plt.figure()
    f.subplots_adjust(wspace=0.4, hspace=0.6)
    ax_dist = f.add_subplot(2,3,1)
    ax_frag = f.add_subplot(2,3,2)
    ax_healthloss = f.add_subplot(2,3,3)
    
    ax_suicide = f.add_subplot(2,3,4)
    ax_total = f.add_subplot(2,3,5)
    ax_dist.set_title("reward_dist")
    ax_frag.set_title("rewad_frag")
    ax_healthloss.set_title("reward_healthloss")
    ax_suicide.set_title("reward_suicide")
    ax_total.set_title("reward_total")
    ax_dist.plot(dist_all)
    ax_frag.plot(frag_all)
    ax_healthloss.plot(healthloss_all)
    ax_suicide.plot(suicide_all)
    ax_total.plot(total)
    return f

def plot_rewards_match(rewards):
    reward_dist = rewards['dist'].values
    reward_frag = rewards['frag'].values
    reward_healthloss = rewards['healthloss'].values
    reward_suicide = rewards['suicide'].values
    reward_total = reward_frag + reward_healthloss + reward_suicide + reward_dist

    f = plt.figure()
    f.subplots_adjust(wspace=0.4, hspace=0.6)
    ax_dist = f.add_subplot(2,3,1)
    ax_frag = f.add_subplot(2,3,2)
    ax_healthloss = f.add_subplot(2,3,3)
    ax_suicide = f.add_subplot(2,3,4)
    ax_total = f.add_subplot(2,3,5)
    ax_dist.set_title("reward_dist")
    ax_frag.set_title("rewad_frag")
    ax_healthloss.set_title("reward_healthloss")
    ax_suicide.set_title("reward_suicide")
    ax_total.set_title("reward_total")
    ax_dist.plot(reward_dist)
    ax_frag.plot(reward_frag)
    ax_healthloss.plot(reward_healthloss)
    ax_suicide.plot(reward_suicide)
    ax_total.plot(reward_total)
    return f

