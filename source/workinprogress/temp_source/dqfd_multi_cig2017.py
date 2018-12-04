#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import multiprocessing
import skimage.color, skimage.transform
from vizdoom import *
import os, time, random, threading, h5py, math
import tensorflow as tf
import numpy as np
from game_instance import GameInstanceBasic, GameInstance
from global_constants import *
from datetime import datetime, timedelta
from PIL import Image
import matplotlib.pyplot as plt
from replay_memory import ReplayMemory


# In[ ]:


LOGDIR = "./logs/log_test/"
MODEL_PATH =  "./models/model_test/model.ckpt"
CONFIG_FILE_PATH = "./config/custom_config.cfg"
WEIGHTS_PATH = ["./weights_merged/conv1_kernel_expand.npy", "./weights_merged/conv1_bias.npy", "./weights_merged/conv2_kernel_expand.npy", "./weights_merged/conv2_bias.npy"]
DEMO_PATH = ["./demonstration/imitation_learning_v3/demodata%02d.hdf5"%(i) for i in [1,2]]
__name__ = "learning_async"
# __name__ = "learning_imitation"
N_ACTION = 6
N_AGENT_ACTION = 2**6
BOTS_NUM = 8
N_WORKERS = 10
REWARDS = {'living':-0.01, 'health_loss':-1, 'medkit':50, 'ammo':0.0, 'frag':500, 'dist':1e-3, 'suicide':-500} 
LSTM_SIZE = 1024
N_ADV = 5
LAMBDA1 = 1.0
LAMBDA2 = 1.0
LAMBDA3 = 0.001
RESOLUTION = (120,120,3)
MERGIN_VALUE = 100.0
INTERVAL_BATCH_LEARNING = 10
INTERVAL_UPDATE_NETWORK = 10
N_BATCH = 64
INTERVAL_UPDATE_ORIGIN = 10
N_WORKERS = 1
np.random.seed(0)


# In[ ]:


for f in os.listdir(LOGDIR):
    os.remove(os.path.join(LOGDIR, f))


# In[ ]:


class Environment(object):
    def __init__(self,sess,  name, game_instance, network, agent, start_time, end_time, random_seed=0):
#     def __init__(self,sess,  name, start_time, end_time, parameter_server):
        self.name = name
        self.sess = sess
        self.game = game_instance
        self.game.game.set_seed(random_seed)
        self.game.game.set_render_weapon(False)
        self.game.game.set_render_crosshair(False)
        self.game.game.init()
        self.network = network
        self.agent = agent
        
        self.clear_obs()
        self.clear_batch()
        
        self.start_time = start_time
        self.end_time = end_time
        self.progress = 0.0
        self.log_server = None
        
        self.replay_memory = None
        
        self.step = 0
        self.model_gen_count = 0
        
        self.times_act = None
        self.times_update = None
        
        self.count_update = 0
        self.rewards_detail = None
        print(self.name," initialized...")
        
    def run_learning(self, coordinator):
        print(self.name + " start learning")
        
        self.game.new_episode(BOTS_NUM)
        try:
            while not coordinator.should_stop():
                self.learning_step()
                self.progress = (datetime.now().timestamp() - self.start_time)/(self.end_time - self.start_time)
                if self.progress >= 1.0:
                    break
        except Exception as e:
            coordinator.request_stop(e)
        return 0
    
    def run_prelearning(self, coordinator):
        assert self.replay_memory is not None
        try:
            while not coordinator.should_stop():
                loss_values = self.prelearning_step()
                self.progress = (datetime.now().timestamp() - self.start_time)/(self.end_time - self.start_time)
                if self.progress >= 1.0:
                    break
        except Exception as e:
            coordinator.request_stop(e)
        return 0
    
    def run_test(self, coordinator):
        
        try:
            while not coordinator.should_stop():
                reward,frag, death,kill,total_detail = self.test_agent()
                if self.rewards_detail is not None:
                    self.rewards_detail.append(total_detail)
                print("----------TEST at %.1f ---------"%(self.progress*100))
                print("FRAG:",frag,"KILL:",kill, "DEATH:",death)
                print("REWARD:",reward)

                if self.log_server is not None:
                    self.log_server.write_score(self.sess,self.step,  reward, frag, death ,kill)
                    if self.progress >= self.model_gen_count/12:
                        self.model_gen_count += 1
                        self.log_server.save_model(sess=self.sess, model_path=MODEL_PATH, step=self.model_gen_count+1)
                    
                self.step += 1
                time.sleep(60)
                self.progress = (datetime.now().timestamp() - self.start_time)/(self.end_time - self.start_time)
                if self.progress >= 1.0:
                    break
        except Exception as e:
            coordinator.request_stop(e)
            
    def learning_step(self):
        self.network.pull_parameter_server(self.sess)
        loss_values = []
        if not self.game.is_episode_finished() and self.game.get_screen_buff() is not None:
            
            if self.times_act is not None:
                start_time = datetime.now().timestamp()

            s1_ = self.preprocess(self.game.get_screen_buff())
            self.push_obs(s1_)
            agent_action_idx = self.agent.act_eps_greedy(self.sess, self.obs['s1'], self.progress)
            engin_action = self.convert_action_agent2engine(agent_action_idx)
            r,r_detail = self.game.make_action(engin_action , FRAME_REPEAT)
            isterminal = self.game.is_episode_finished()
            if isterminal:
                s2_ = np.zeros(RESOLUTION)
            else:
                s2_ = self.preprocess(self.game.get_screen_buff())
            
            self.push_batch( s1_, agent_action_idx, s2_, r , isterminal, False)
            
            if self.times_act is not None:
                self.times_act.append(datetime.now().timestamp() - start_time)
            
            if len(self.memory) >= N_ADV or isterminal:
                batch = self.make_advantage_data()
                for i,b in enumerate(batch):
                    if len(b) == 8:
                        self.replay_memory.store(b)

            if (self.game.is_player_dead()):
                self.game.respawn_player()
            
            self.step += 1
            
            if self.step % INTERVAL_UPDATE_NETWORK == 0:
                self.network.copy_network_learning2target(self.sess)
                
            if self.times_update is not None:
                start_time = datetime.now().timestamp()
            
            if self.step % INTERVAL_BATCH_LEARNING == 0 and len(self.replay_memory) >= N_BATCH:
                s1, actions, r_one, r_adv, isdemo, is_weight, tree_idx = self.make_batch()
                loss_values = self.network.update_parameter_server(self.sess, s1, actions, r_one, r_adv, isdemo, is_weight)
                self.count_update += 1
                tderror = loss_values[4]
                loss_values = loss_values[:-1]
                self.replay_memory.batch_update(tree_idx, tderror)
                if self.log_server is not None:
                    l_one, l_n, l_m = np.mean(loss_values[:-1], axis=1)
                    l_l = loss_values[-1]
                    self.log_server.write_loss(self.sess,self.step ,l_one, l_n, l_m, l_l)
                    self.log_server.write_img(self.sess, self.step, s1[0:1])
                    self.log_server.write_weights(self.sess, self.step)
                    
            if self.times_update is not None:
                self.times_update.append(datetime.now().timestamp() - start_time)
        else:
            self.game.new_episode(BOTS_NUM)
            self.clear_batch()
            self.clear_obs()

        return loss_values
    
    def prelearning_step(self):
        self.network.pull_parameter_server(self.sess)

        s1, actions, r_one, r_adv, isdemo, is_weight, tree_idx = self.make_batch()
        loss_values = self.network.update_parameter_server(self.sess, s1, actions, r_one, r_adv, isdemo, is_weight)
        tderror = loss_values[4]
        loss_values = loss_values[:-1]
        print(is_weight)
        self.replay_memory.batch_update(tree_idx, tderror) 
        
        if self.step % INTERVAL_UPDATE_NETWORK == 0:
            self.network.copy_network_learning2target(self.sess)
        
        if self.log_server is not None:
            if self.step % 10 == 0:
                l_one, l_n, l_m = np.mean(loss_values[:-1], axis=1)
                l_l = loss_values[-1]
                self.log_server.write_loss(self.sess, self.step, l_one, l_n, l_m, l_l)
                self.log_server.write_img(self.sess, self.step, s1[0:1])
                self.log_server.write_weights(self.sess, self.step)
        self.step += 1
        return loss_values
            
    def test_agent(self, gif_buff=None, reward_buff=None, sample_imgs=None):
        
        self.game.new_episode(BOTS_NUM)
        
#         Copy params from global
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
            engine_action = self.convert_action_agent2engine(action)
            reward,reward_detail = self.game.make_action(engine_action,FRAME_REPEAT)
            isterminal = self.game.is_episode_finished()
            total_reward += reward
            for k in reward_detail.keys():
                if not k in total_detail.keys():
                    total_detail[k] = reward_detail[k]
                else:
                    total_detail[k] += reward_detail[k]
            step += 1
            if reward_buff is not None:
                reward_buff.append((engine_action, reward))
            
            if (self.game.is_player_dead()):
                self.game.respawn_player()
        
        save_img = []
        if gif_buff is not None:
            for i in range(len(gif_img)):
                save_img.append(Image.fromarray(np.uint8(gif_img[i])))
            gif_buff += save_img
        return total_reward, self.game.get_frag_count(), self.game.get_death_count(), self.game.get_kill_count(), total_detail
        
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
        for i in range(6):
            ans.append(agent_action%2)
            agent_action = int(agent_action / 2)
        return ans
    
    def preprocess(self,img):
        if len(img.shape) == 3 and img.shape[0]==3:
            img = img.transpose(1,2,0)

        img = skimage.transform.resize(img, RESOLUTION, mode="constant")
        img = img.astype(np.float32)
#         img = (img)/255.0
        return img

    def push_obs(self, s1):
        self.obs['s1'] = np.roll(self.obs['s1'],shift=-1, axis=0)
        self.obs['s1'][-1] = s1
        
    def clear_obs(self):
        self.obs = {}
        self.obs['s1'] = np.zeros((N_ADV,)+ RESOLUTION, dtype=np.float32)
        
    def push_batch(self, s1, action,s2,  reward, isterminal, isdemo):
        self.states_buff = np.roll(self.states_buff, shift= -1, axis=0)
        self.states_buff[-1] = s1
        self.memory.append([np.copy(self.states_buff), action, np.copy(np.concatenate([self.states_buff[1:], [s2]], axis=0)) , reward, isterminal, isdemo])
    
    def clear_batch(self):
        self.states_buff = np.zeros((N_ADV,) + RESOLUTION, dtype=np.float32)
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
        tree_idx, batch_row, is_weight = self.replay_memory.sample(N_BATCH, self.agent.calc_eps(self.progress))
        s1, actions, s2, r_one, r_adv, isdemo = [],[],[],[],[],[]
        
        s2_adv = [ batch_row[i,3] for i in range(N_BATCH)]
        predicted_q_adv  = self.network.get_qvalue_target_max(self.sess,s2_adv)
        
        s2 = [ batch_row[i,2] for i in range(N_BATCH)]
        predicted_q = self.network.get_qvalue_target_max(self.sess,s2)
        
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
        


# In[ ]:


class ParameterServer:
    def __init__(self, sess, log_dir):
        self.sess = sess
        with tf.variable_scope("parameter_server", reuse=tf.AUTO_REUSE):
            self.state1_ = tf.placeholder(tf.float32, shape=(None,N_ADV,) + RESOLUTION)
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
            maxpool1 = NetworkSetting.maxpool1(conv1)
            conv2 = NetworkSetting.conv2(maxpool1)
            maxpool2 = NetworkSetting.maxpool2(conv2)
            reshape = NetworkSetting.reshape(maxpool2)
            rnn = NetworkSetting.lstm(reshape, state)
            fc1 = NetworkSetting.fc1(rnn)
            q = NetworkSetting.q_value(fc1)
            
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
        self.loss_one_ = tf.placeholder(tf.float32, shape=(), name="loss_one")
        self.loss_adv_ = tf.placeholder(tf.float32, shape=(), name="loss_adv")
        self.loss_cls_ = tf.placeholder(tf.float32, shape=(), name="loss_class")
        self.loss_l2_ = tf.placeholder(tf.float32, shape=(), name="loss_l2")
        
        with tf.variable_scope("Summary_Score"):
            s = [tf.summary.scalar('reward', self.reward_, family="score"), tf.summary.scalar('frag', self.frag_, family="score"), tf.summary.scalar("death", self.death_, family="score"), tf.summary.scalar("kill", self.kill_, family="score")]
            self.summary_reward = tf.summary.merge(s)
        
        with tf.variable_scope("Summary_Loss"):
            self.summary_loss = tf.summary.merge([tf.summary.scalar('loss_one', self.loss_one_, family="loss"), tf.summary.scalar('loss_adv', self.loss_adv_, family="loss"),                                                   tf.summary.scalar('loss_class', self.loss_cls_, family="loss"), tf.summary.scalar('loss_l2', self.loss_l2_, family='loss')])
        
        with tf.variable_scope("Summary_Images"):
            conv1_display = tf.reshape(tf.transpose(self.conv1, [0,1,4,2,3]), (-1, self.conv1.get_shape()[1],self.conv1.get_shape()[2]))
            conv2_display = tf.reshape(tf.transpose(self.conv2, [0,1,4,2,3]), (-1, self.conv2.get_shape()[1],self.conv2.get_shape()[2]))
            conv1_display = tf.expand_dims(conv1_display, -1)
            conv2_display = tf.expand_dims(conv2_display, -1)

            state_shape = self.state1_.get_shape()
            conv1_shape = conv1_display.get_shape()
            conv2_shape = conv2_display.get_shape()

            s_img = []
            s_img.append(tf.summary.image('state',tf.reshape(self.state1_,[-1, state_shape[2], state_shape[3], state_shape[4]]), 1, family="state1"))
            s_img.append(tf.summary.image('conv1',tf.reshape(self.conv1,[-1, conv1_shape[1], conv1_shape[2], 1]), family="conv1"))
            s_img.append(tf.summary.image('conv2',tf.reshape(self.conv2,[-1, conv2_shape[1], conv2_shape[2], 1]), family="conv2"))

            self.summary_image = tf.summary.merge(s_img)
            
        with tf.variable_scope("Summary_Weights"):
            s = [tf.summary.histogram(values=w, name=w.name, family="weights") for w in self.weights_params]
            self.summary_weights = tf.summary.merge(s)

        self.writer = tf.summary.FileWriter(log_dir)
        
    def write_graph(self, sess):
        self.writer.add_graph(sess.graph)
        
    def write_score(self,sess, step ,reward, frag, death, kill):
        m = sess.run(self.summary_reward, feed_dict={self.reward_:reward, self.frag_:frag, self.death_:death, self.kill_:kill})
        return self.writer.add_summary(m, step)
    
    def write_loss(self,sess, step, l_o, l_n,l_c, l_l):
        m = sess.run(self.summary_loss, feed_dict={self.loss_one_: l_o, self.loss_adv_:l_n, self.loss_cls_:l_c, self.loss_l2_:l_l})
        return self.writer.add_summary(m, step)
    
    def write_img(self,sess, step, state):
        m = sess.run(self.summary_image, feed_dict={self.state1_: state})
        return self.writer.add_summary(m, step)
    
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
    
    def __init__(self, network):
        self.network = network
        
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
        if random.random() <= eps:
#             p = self.network.get_policy(sess, [s1])[0]
#             a_idx = np.random.choice(N_AGENT_ACTION, p=p)
            a_idx = self.network.get_best_action(sess, [s1])[0]
#             a_idx = self.get_sum_prob(sess, s1)
        else:
            a_idx = np.random.randint(N_AGENT_ACTION)
            
        return a_idx
    
    def act_greedy(self, sess, s1):
#         p = self.network.get_policy(sess, [s1])[0]
#         a_idx = np.random.choice(N_AGENT_ACTION, p=p)
        a_idx = self.network.get_best_action(sess, [s1])[0]
#         a_idx = self.get_sum_prob(sess, s1)
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
                self.state1_ = tf.placeholder(tf.float32,shape=(None,N_ADV)+RESOLUTION, name="state_1")
                self.q_value, self.conv1, self.conv2 = self._build_model(self.state1_)
            with tf.variable_scope("target_network"):
                self.state1_target_ = tf.placeholder(tf.float32,shape=(None,N_ADV)+RESOLUTION, name="state_1")
                self.q_value_target,_,_ = self._build_model(self.state1_target_)
            
            self.a_ = tf.placeholder(tf.int32, shape=(None,), name="action")
            self.target_one_ = tf.placeholder(tf.float32, shape=(None,), name="target_one_")
            self.target_n_ = tf.placeholder(tf.float32, shape=(None,), name="target_n_")
            self.isdemo_ = tf.placeholder(tf.float32,shape=(None,), name="isdemo_")
            self.mergin_ = tf.placeholder(tf.float32,shape=(None,N_AGENT_ACTION), name="mergin_")
            self.is_weight_ = tf.placeholder(tf.float32, shape=(None,), name="is_weight")
                
            self._build_graph()

        self.update_global_weight_params =             parameter_server.optimizer.apply_gradients(zip(self.grads, parameter_server.weights_params))

    def _build_model(self,state):
        conv1 = NetworkSetting.conv1(state)
        maxpool1 = NetworkSetting.maxpool1(conv1)
        conv2 = NetworkSetting.conv2(maxpool1)
        maxpool2 = NetworkSetting.maxpool2(conv2)
        reshape = NetworkSetting.reshape(maxpool2)
        rnn  = NetworkSetting.lstm(reshape, state)
        fc1 = NetworkSetting.fc1(rnn)
        
        q_value = NetworkSetting.q_value(fc1)
        
        return q_value, conv1, conv2

    def _build_graph(self):

        self.q_prob = tf.nn.softmax(self.q_value)
        self.q_argmax = tf.argmax(self.q_value, axis=1)
        self.q_learning_max = tf.reduce_max(self.q_value, axis=1)
        self.q_target_max = tf.reduce_max(self.q_value_target, axis=1)
        
        self.weights_params_learning = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+"/learning_network")
        self.weights_params_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+"/target_network")
        
        self.tderror_one = tf.abs(self.target_one_ - tf.reduce_max(self.q_value, axis=1))
        self.loss_one = tf.square(self.tderror_one)
        self.tderror_n = tf.abs(self.target_n_ - tf.reduce_max(self.q_value, axis=1))
        self.loss_n = LAMBDA1 * tf.square(self.tderror_n)
        self.loss_l2 = LAMBDA3 * tf.reduce_sum([tf.nn.l2_loss(w) for w in self.weights_params_learning])
        
        idx = tf.transpose([tf.range(tf.shape(self.q_value)[0]), self.a_])
        self.loss_mergin = LAMBDA2 * ((tf.stop_gradient(tf.reduce_max(self.q_value + self.mergin_, axis=1)) - tf.gather_nd(self.q_value,indices=idx))*self.isdemo_)
        
        self.tderror_total = self.tderror_one + self.tderror_n + self.loss_mergin
        self.loss_total = tf.reduce_mean(self.loss_one +  self.loss_n + self.loss_mergin + self.loss_l2) * self.is_weight_
        
        self.grads = tf.gradients(self.loss_total ,self.weights_params_learning)
        
        self.pull_global_weight_params = [l_p.assign(g_p) for l_p,g_p in zip(self.weights_params_learning,parameter_server.weights_params)]
        self.push_local_weight_params = [g_p.assign(l_p) for g_p,l_p in zip(parameter_server.weights_params,self.weights_params_learning)]
        
        self.copy_params = [t.assign(l) for l,t in zip(self.weights_params_learning, self.weights_params_target)]
        
    def copy_network_learning2target(self, sess):
        return sess.run(self.copy_params)
        
    def pull_parameter_server(self, sess):
        return sess.run(self.pull_global_weight_params)
    
    def push_parameter_serverl(self):
        sess.run(self.push_local_weight_params)
        
    def get_weights_learngin(self, sess):
        return sess.run(self.weights_params_learning)
    
    def get_weights_target(self, sess):
        return sess.run(self.weights_params_target)
    
    def update_parameter_server(self, sess, s1, a, target_one,target_n, isdemo, is_weight):
        assert np.ndim(s1) == 5
        mergin_value = np.ones((len(s1), N_AGENT_ACTION)) * MERGIN_VALUE
        mergin_value[range(len(s1)), a] = 0.0
        feed_dict = {self.state1_: s1,self.a_:a, self.target_one_:target_one, self.target_n_:target_n, self.isdemo_:isdemo, self.is_weight_:is_weight, self.mergin_:mergin_value}
        _,l_one, l_n, l_mergin, l_l2, tderror_total = sess.run([self.update_global_weight_params, self.loss_one, self.loss_n, self.loss_mergin, self.loss_l2, self.tderror_total], feed_dict)
        return l_one, l_n, l_mergin, l_l2, tderror_total
    
    def check_weights(self, sess):
        weights = SESS.run(self.weights_params_learning)
        assert np.isnan([np.mean(w) for w in weights]).any()==False , print(weights)

    def get_qvalue_learning(self, sess, s1):
        assert np.ndim(s1) == 5
        return sess.run(self.q_value, {self.state1_: s1})
    
    def get_qvalue_lerning_max(self, sess, s1):
        return sess.run(self.q_learing_max, {self.state1_:s1})

    def get_qvalue_target(self, sess ,s1):
        assert np.ndim(s1) == 5
        return sess.run(self.q_value_target, {self.state1_target_:s1})
    
    def get_qvalue_target_max(self, sess, s1):
        return sess.run(self.q_target_max, {self.state1_target_:s1})
    
    def get_policy(self, sess, s1):
        return sess.run(self.q_prob, {self.state1_: s1})
    
    def get_best_action(self,sess, s1):
        return sess.run(self.q_argmax, {self.state1_:s1})


# In[ ]:


class NetworkSetting:
    
    def conv1(pre_layer):
        num_outputs = 32
        kernel_size = [1,6,6]
        stride = [1,3,3]
        padding = 'SAME'
        activation = tf.nn.relu
        weights_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(0.1)
        return tf.layers.conv3d(pre_layer,kernel_size=kernel_size,                                        filters=num_outputs,                                        strides=stride,padding=padding,activation=activation,                                        kernel_initializer=weights_init,                                        bias_initializer=bias_init)
    
    def maxpool1(pre_layer):
        return tf.nn.max_pool3d(pre_layer,[1,1,3,3,1],[1,1,2,2,1],'SAME')
    
    def conv2(pre_layer):
        num_outputs = 32
        kernel_size = [1,3,3]
        stride = [1,2,2]
        padding = 'SAME'
        activation = tf.nn.relu
        weights_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(0.1)
        return tf.layers.conv3d(pre_layer,kernel_size=kernel_size,filters=num_outputs,                                        strides=stride,padding=padding,activation=activation,                                        kernel_initializer=weights_init,bias_initializer=bias_init)
    
    def maxpool2(pre_layer):
        return tf.nn.max_pool3d(pre_layer,[1,1,3,3,1],[1,1,2,2,1],'SAME')
        
    def reshape(pre_layer):
        shape = pre_layer.get_shape()
        return tf.reshape(pre_layer, shape=(-1,shape[1], shape[2]*shape[3]*shape[4]))
    
    def lstm(pre_layer, state):
        batch_size = tf.shape(pre_layer)[0]
        cell = tf.nn.rnn_cell.LSTMCell(LSTM_SIZE)
        rnn_state = cell.zero_state(batch_size, dtype=tf.float32)
        rnn_out, state_out = tf.nn.dynamic_rnn(cell, pre_layer, initial_state=rnn_state, dtype=tf.float32)
        return rnn_out[:,-1]
    
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


def load_demo(replay_memory, demo_path):
    for demo in demo_path[:]:
        print(demo)
        file = h5py.File(demo, 'r')
        episodes = list(file.keys())[1:]
        game = GameInstance(DoomGame(), name="noname", rewards =REWARDS, config_file_path=CONFIG_FILE_PATH,n_adv=INTERVAL_UPDATE_ORIGIN)
        total_n_transit = 0
        for e in episodes:
            states_row = file[e+"/states"][:]
            action_row = file[e+"/action"][:]
            ammo = file[e+"/ammo"][:]
            frag = file[e+"/frag"][:]
            health = file[e+"/health"][:]
            posx = file[e+"/posx"][:]
            posy = file[e+"/posy"][:]
            
            n_transit = len(states_row)
            total_n_transit += n_transit
            
            actions = []
            for a_r in action_row:
                ans = 0
                for i, e_a in enumerate(a_r):
                    ans += e_a * 2**i
                actions.append(ans)
            actions = np.array(actions)
            
            pre_frag, pre_health, originx, originy, pre_ammo = frag[0], health[0], posx[0], posy[0], ammo[0]
            memory = []
            states_buff = np.zeros((N_ADV,) + RESOLUTION)
            
            for i in range(0, n_transit):
                s1_ = states_row[i]
                if i == n_transit -1 :
                    isterminal = True
                    s2_ = np.zeros(RESOLUTION)
                    r = 0
                else:
                    isterminal = False
                    s2_ = states_row[i+1]
                    r,r_detail = game.get_reward(frag[i+1]-pre_frag, 0, health[i+1] - pre_health, ammo[i+1] - pre_ammo, posx[i+1] - originx, posy[i+1]-originy)
                
                states_buff = np.roll(states_buff, shift= -1, axis=0)
                states_buff[-1] = s1_
                
                memory.append([np.copy(states_buff),actions[i], np.copy(np.concatenate([states_buff[1:], [s2_]], axis=0)), r, isterminal, True])

                if(i%INTERVAL_UPDATE_ORIGIN == 0):
                    originx, originy = posx[i],posy[i]

                pre_frag, pre_health, pre_ammo = frag[i], health[i], ammo[i]
                
                if len(memory) == N_ADV or i == n_transit-1:
                    R_adv = 0
                    len_memory = len(memory)
                    _, _, s2_adv, _, _, _ = memory[-1]
                    for i in range(len_memory - 1, -1, -1):
                        s1,a, s2 ,r,isterminal,isdemo = memory[i]
                        R_adv = r + GAMMA*R_adv
                        replaymemory.store(np.array([s1, a,s2 ,s2_adv,r ,R_adv ,isterminal, isdemo]))
                    memory = []                    
                
    replay_memory.tree.set_parmanent_data(total_n_transit)
    print(len(replay_memory), "data are stored")
    return 0


# In[ ]:


if __name__=="learning_imitation":
    replaymemory = ReplayMemory(10000)
    rewards = load_demo(replaymemory, DEMO_PATH)
    
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list=USED_GPU))
    config.log_device_placement = False
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    with tf.device('/gpu:0'):
        parameter_server = ParameterServer(sess,LOGDIR)

        starttime = datetime.now().timestamp()
        end_time = (datetime.now() + timedelta(minutes=30)).timestamp()
        
        coordinator = tf.train.Coordinator()

        name = "worker_imitation"
        game_instance=GameInstance(DoomGame(), name=name, rewards=REWARDS, config_file_path=CONFIG_FILE_PATH,n_adv=INTERVAL_UPDATE_ORIGIN)
        network = NetworkLocal(name, parameter_server)
        agent = Agent(network)
        imitation_env = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, start_time=starttime, end_time=end_time, random_seed=0)
        imitation_env.log_server = parameter_server
        imitation_env.replay_memory = replaymemory
        thread_imitation = threading.Thread(target=imitation_env.run_prelearning, args=(coordinator,))

        name = "test"
        game_instance=GameInstance(DoomGame(), name=name, rewards=REWARDS, config_file_path=CONFIG_FILE_PATH, n_adv=INTERVAL_UPDATE_ORIGIN)
        game_instance.something = []
        network = NetworkLocal(name, parameter_server)
        agent = Agent(network)
        test_env = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, start_time=starttime, end_time=end_time, random_seed=0)
        test_env.log_server = parameter_server
        thread_test = threading.Thread(target=test_env.run_test, args=(coordinator,))
        
        parameter_server.write_graph(sess)
        sess.run(tf.global_variables_initializer())


# In[ ]:


print("-----Start IMITATION LEARNING----")
threads = [thread_imitation, thread_test]
for t in threads:
    t.start()
coordinator.join(threads)

parameter_server.save_model(sess=sess, step=15, model_path=MODEL_PATH)

GIF_BUFF = []
REWARD_BUFF = []
r,f,d,imgs = test_env.test_agent(gif_buff=GIF_BUFF,reward_buff=REWARD_BUFF)
GIF_BUFF[0].save('gifs/test.gif',save_all=True, append_images=GIF_BUFF[1:], optimize=False, duration=40*4, loop=0)


# In[ ]:


N_WORKERS = 10


# In[ ]:


if __name__=="learning_async":
    replaymemory = ReplayMemory(10000)
    rewards = load_demo(replaymemory, DEMO_PATH)
    
    tf.set_random_seed(0)
    
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list=USED_GPU))
    config.log_device_placement = False
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    with tf.device('/gpu:0'):
        parameter_server = ParameterServer(sess,LOGDIR)

        starttime = datetime.now().timestamp()
        end_time = (datetime.now() + timedelta(minutes=60)).timestamp()
        
        coordinator = tf.train.Coordinator()
        
        environments, threads = [], []
        
        for i in range(N_WORKERS):
            name = "worker_%d"%(i+1)
            game_instance=GameInstance(DoomGame(), name=name, rewards=REWARDS, config_file_path=CONFIG_FILE_PATH,n_adv=INTERVAL_UPDATE_ORIGIN)
            network = NetworkLocal(name, parameter_server)
            agent = Agent(network)
            e = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, start_time=starttime, end_time=end_time, random_seed=i)
            e.replay_memory = replaymemory
            environments.append(e)

        environments[0].log_server = parameter_server
        environments[0].times_act = []
        environments[0].times_update = []

        name = "test"
        game_instance=GameInstance(DoomGame(), name=name, rewards=REWARDS, config_file_path=CONFIG_FILE_PATH, n_adv=INTERVAL_UPDATE_ORIGIN)
        network = NetworkLocal(name, parameter_server)
        agent = Agent(network)
        test_env = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, start_time=starttime, end_time=end_time, random_seed=0)
        test_env.log_server = parameter_server
        test_env.rewards_detail = []
        thread_test = threading.Thread(target=test_env.run_test, args=(coordinator,))
        
        for e in environments:
            threads.append(threading.Thread(target=e.run_learning, args=(coordinator,)))
        
        threads.append(thread_test)
        
        parameter_server.write_graph(sess)
        sess.run(tf.global_variables_initializer())
        


# In[ ]:


parameter_server.load_model(sess=sess, step=15, model_path="./models/dqfd_imitationonly/model_imitationonly_181123/model.ckpt")

print("-----Start ASYNC LEARNING----")
for t in threads:
    t.start()
coordinator.join(threads)

parameter_server.save_model(sess=sess, step=15, model_path=MODEL_PATH)

GIF_BUFF = []
REWARD_BUFF = []
r,f,d,imgs,_ = test_env.test_agent(gif_buff=GIF_BUFF,reward_buff=REWARD_BUFF)
GIF_BUFF[0].save('gifs/test.gif',save_all=True, append_images=GIF_BUFF[1:], optimize=False, duration=40*4, loop=0)


# In[ ]:


parameter_server.load_model(sess = sess, step=15, model_path="./data_results/dqfd_cig2017_single/model_test/model.ckpt")


# In[ ]:


len(test_env.game.something)


# In[ ]:


distances = []
frags = []
for i in range(50):
    test_env.game.something = []
    r,f,d,kill,total_detail = test_env.test_agent()
    frags.append(f)
    distances.append(test_env.game.something)


# In[ ]:


np.mean(frags_n1)


# In[ ]:


np.var(frags_n1)


# In[ ]:


frags_n1 = frags


# In[ ]:


print(frags_n1)


# In[ ]:


frags_n1 = np.array(frags_n1)


# In[ ]:


frags_n10 = np.array(frags_n10)


# In[ ]:


suicide_n10 = np.minimum(frags_n10,0)
suicide_n1= np.minimum(frags_n1,0)


# In[ ]:


np.mean(suicide_n1)


# In[ ]:


np.var(suicide_n1)


# In[ ]:


np.mean(suicide_n10)


# In[ ]:


np.var(suicide_n10)


# In[ ]:


np.var(frags_n10)


# In[ ]:


np.mean(frags_n10)


# In[ ]:


sum(frags_n1 > 0)


# In[ ]:


len(distances)


# In[ ]:


distances = np.array(distances)
np.save("./data_results/dqfd_cig2017_single/distances_async_learned.npy",distances)


# In[ ]:


np.save("./data_results/dqfd_cig2017_10/distances_async_learned.npy", distances)


# In[ ]:


distances_2d = distances

distance_1d = []
for dist in distances_2d:
    d_tmp = []
    for d in dist:
        d_tmp.append(np.sqrt(d[0]**2 + d[1]**2))
    distance_1d.append(sum(d_tmp))
    
distance_1d = np.array(distance_1d)

distance_1d.mean()


# In[ ]:


distanes = np.load("./data_results/dqfd_cig2017_10/distances_async_learned.npy")


# In[ ]:


distances[0]


# In[ ]:


times_act = environments[0].times_act

time_update = environments[0].times_update

times_update = np.array(time_update)

times_act = np.array(times_act)

plt.plot(times_act)


# In[ ]:


plt.plot(times_update[:])


# In[ ]:


times_act.mean()


# In[ ]:


times_update[times_update>0.15].mean()


# In[ ]:


np.save("./data_results/dqfd_cig2017_10/times_act.npy", times_act)


# In[ ]:


np.save("./data_results/dqfd_cig2017_10/times_update.npy", time_update)


# In[ ]:


sum([e.count_update for e in environments])


# In[ ]:


rewards_distance = []
rewards_total = []
for r_d in test_env.rewards_detail:
    rewards_distance.append(r_d['dist'])
    rewards_total.append(sum(r_d.values()))
    
rewards_total = np.array(rewards_total)
rewards_distance = np.array(rewards_distance)


# In[ ]:


np.save("./data_results/dqfd_cig2017_10/rewards_total.npy", rewards_total)
np.save("./data_results/dqfd_cig2017_10/rewards_detail.npy", rewards_distance)


# In[ ]:


plt.plot(rewards_total)


# In[ ]:


plt.plot(rewards_distance)


# In[ ]:


rewards_dist1 = np.load("data_results/dqfd_cig2017_single/rewards_detail.npy")
rewards_dist10 = np.load("data_results/dqfd_cig2017_10/rewards_detail.npy")


# In[ ]:


f = plt.figure()
ax = f.add_subplot(1,1,1)
x = np.arange(rewards_dist1.shape[0])/ rewards_dist1.shape[0]
ax.plot(x, rewards_dist1, label='Training Data')
x = np.arange(rewards_dist10.shape[0])/ rewards_dist10.shape[0]
ax.plot(x,rewards_dist10, label='Test Data')
# ax.legend()
# f.savefig('predict_risisuicide_acc.pdf')


# In[ ]:





# In[ ]:


sum(test_env.rewards_detail[0].values())


# In[ ]:


if __name__ == "test":
    MODEL_PATH = "./models/dqfd_imitationonly/model_imitationonly_181117/"
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
        game_instance=GameInstance(DoomGame(), name=name, config_file_path=CONFIG_FILE_PATH,rewards=REWARDS, n_adv=N_ADV)
        network = NetworkLocal(name, parameter_server)
        agent = Agent(network)
        test_env = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, start_time=starttime, end_time=end_time)


# In[ ]:


parameter_server.load_model(sess=sess, model_path=MODEL_PATH, step=2)


# In[ ]:


GIF_BUFF = []
SAMPLE_BUFF = []
test_env.test_agent(gif_buff=GIF_BUFF,sample_imgs=SAMPLE_BUFF)


# In[ ]:





# In[ ]:


IDX = 0
Q = []


# In[ ]:


q_ = test_env.network.get_qvalue_learning(sess=sess, s1=[sample_buff[IDX:IDX+5]])
plt.bar(range(N_AGENT_ACTION),q_[0,:])
Q.append(q_)
IDX += 1


# In[ ]:


for i,q in enumerate(Q):
    print(Q[i]-Q[i+1])


# In[ ]:


q_ = test_env.network.get_qvalue_target(sess=sess, s1=[sample_buff[0:5]])
plt.bar(range(N_AGENT_ACTION),q_[0,:])


# In[ ]:


q_.shape


# In[ ]:


plt.bar(range(N_AGENT_ACTION),q_[0,:])


# In[ ]:


GIF_BUFF[0].save('gifs/test_.gif',save_all=True, append_images=GIF_BUFF[1:], optimize=False, duration=40*4, loop=0)


# In[ ]:


if __name__ == "test_game_instance":
    environments[0].game.new_episode(BOTS_NUM)
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

