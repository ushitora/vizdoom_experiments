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
DEMO_PATH = ["./demonstration/imitation_learning_v3/demodata01.hdf5", "./demonstration/imitation_learning_v3/demodata02.hdf5"]
__name__ = "learning"
N_ACTION = 6
N_AGENT_ACTION = 2 ** N_ACTION
BOTS_NUM = 10
N_WORKERS = 30
REWARDS = {'living':-0.01, 'health_loss':-1, 'medkit':50, 'ammo':0.0, 'frag':500, 'dist':3e-1, 'suicide':-500}
LSTM_SIZE = 1024


# In[ ]:


for f in os.listdir(LOGDIR):
    os.remove(os.path.join(LOGDIR, f))


# In[ ]:


class Environment(object):
    def __init__(self,sess,  name, game_instance, network, agent, start_time, end_time):
#     def __init__(self,sess,  name, start_time, end_time, parameter_server):
        self.name = name
        self.sess = sess
        self.game = game_instance
        self.network = network
        self.agent = agent
        
        self.start_time = start_time
        self.end_time = end_time
        self.progress = 0.0
        self.log_server = None
        
        self.replay_memory = None
        
        self.step = 0
        self.model_gen_count = 0
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
#                     coordinator.request_stop()
        except Exception as e:
            coordinator.request_stop(e)
        return 0
    
    def run_prelearning(self, coordinator):
        assert self.replay_memory is not None
        try:
            while not coordinator.should_stop():
                l_p, l_v, l_c = self.prelearning_step()
                self.progress = (datetime.now().timestamp() - self.start_time)/(self.end_time - self.start_time)
                if self.progress >= 1.0:
                    break
        except Exception as e:
            coordinator.request_stop(e)
        return 0
    
    def run_test(self, coordinator):
        
        try:
            while not coordinator.should_stop():
                reward,frag, death,_ = self.test_agent()
                print("----------TEST at",(datetime.now()), "---------")
                print("FRAG:",frag, "DEATH:",death)
                print("REWARD",reward)

                if self.log_server is not None:
                    self.log_server.write_score(self.sess,self.step,  reward, frag, death)
                    if self.progress >= self.model_gen_count/12:
                        self.model_gen_count += 1
                        self.log_server.save_model(sess=self.sess, model_path=MODEL_PATH, step=self.model_gen_count+1)
                    
                self.step += 1
                time.sleep(60 * 5)
                self.progress = (datetime.now().timestamp() - self.start_time)/(self.end_time - self.start_time)
                if self.progress >= 1.0:
                    break
#                     coordinator.request_stop()
        except Exception as e:
            coordinator.request_stop(e)
    
    def learning_step(self):
        l_p = 0
        l_v = 0
        self.network.pull_parameter_server(self.sess)
        if not self.game.is_episode_finished() and self.game.get_screen_buff() is not None:

            s1_ = self.preprocess(self.game.get_screen_buff())
            agent_action_idx = self.agent.act_eps_greedy(self.sess, s1_, self.progress)
            engin_action = self.convert_action_agent2engine(agent_action_idx)
            r,r_detail = self.game.make_action(engin_action , FRAME_REPEAT)
            isterminal = self.game.is_episode_finished()

            self.agent.push_batch(s1_, agent_action_idx, r, isterminal)

            if (self.game.is_player_dead()):
                    self.game.respawn_player()

            if self.agent.is_trainable():
                l_p, l_v = self.agent.train_network(self.sess)
                if self.log_server is not None:
                    if self.step % 10 == 0:
                        self.log_server.write_loss(self.sess,self.step, l_p[0], l_v[0])
                        self.log_server.write_weights(self.sess, self.step)
            
            self.step += 1
                
        else:
            self.game.new_episode(BOTS_NUM)
            self.agent.clear_batch()
            self.agent.clear_obs()
            
        return l_p, l_v
    
    def prelearning_step(self):
        self.network.pull_parameter_server_all(self.sess)
        states, actions, rewards, isterminals = self.replay_memory.sample_uniform(BATCH_SIZE)
        l_p, l_v, l_c = self.network.update_parameter_server_imitation(self.sess,states, actions, rewards, isterminals)
        if self.log_server is not None:
            if self.step % 10 == 0:
                self.log_server.write_loss(self.sess, self.step, l_p[0], l_v[0], l_c[0])
                self.log_server.write_weights(self.sess, self.step)
        self.step += 1
        return l_p, l_v, l_c
            
    def test_agent(self, gif_buff=None, reward_buff=None):
        
        self.game.new_episode(BOTS_NUM)
        
#         Copy params from global
        self.network.pull_parameter_server_all(self.sess)

        step = 0
        gif_img = []
        total_reward = 0
        total_detail = {}
        while not self.game.is_episode_finished():
            s1_row = self.game.get_screen_buff()
            s1 = self.preprocess(s1_row)
            if gif_buff is not None:
                gif_img.append(s1_row.transpose(1,2,0))
            action = self.agent.act_greedy(self.sess,s1)
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
        return total_reward, self.game.get_frag_count(), self.game.get_death_count(), gif_img
        
    def convert_action_engine2agent(self,engine_action):
#         return engine_action.index(1)
        assert type(engine_action) == type(list()), print("type: ", type(engine_action))
        ans = 0
        for i, e_a in enumerate(engine_action):
            ans += e_a * 2**i
        return ans
    
    def convert_action_agent2engine(self,agent_action):
#         ans = [0 for i in range(3)]
#         ans[agent_action] = 1
#         return ans
        assert type(agent_action) == type(int()) or type(agent_action) == type(np.int64()), print("type(agent_action)=",type(agent_action))
        ans = []
        for i in range(6):
            ans.append(agent_action%2)
            agent_action = int(agent_action / 2)
        return ans
    
    def add_buff(self, s1,a,r,isterminal):
        self.obs[0][self.buff_pointer] = s1
        self.obs[1][self.buff_pointer] = a
        self.obs[2][self.buff_pointer] = r
        self.obs[3][self.buff_pointer] = isterminal
        self.buff_pointer += 1
        return 0
    
    def clear_buff(self):
        self.obs = [np.zeros(shape=(self.n_adv,) + RESOLUTION, dtype=np.float32),                          np.zeros(shape=(self.n_adv,), dtype=np.int8),                          np.zeros(shape=(self.n_adv,), dtype=np.float32),                          np.ones(shape=(self.n_adv,), dtype=np.int8)]
        self.buff_pointer = 0
        return 0
    
    def preprocess(self,img):
        if len(img.shape) == 3 and img.shape[0]==3:
            img = img.transpose(1,2,0)

        img = skimage.transform.resize(img, RESOLUTION, mode="constant")
        img = img.astype(np.float32)
#         img = (img)/255.0
        return img


# In[ ]:


class ParameterServer:
    def __init__(self, sess, log_dir):
        self.sess = sess
        with tf.variable_scope("parameter_server", reuse=tf.AUTO_REUSE):
            self.state1_ = tf.placeholder(tf.float32, shape=(None,N_ADV,) + RESOLUTION)
            self._build_model(self.state1_)

        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="parameter_server")
        self.weights_params_fc = self.weights_params[4:]
#         self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, RMSProbDecaly)
        self.optimizer = tf.train.AdamOptimizer()
            
        with tf.variable_scope("summary", reuse=tf.AUTO_REUSE):
            self._build_summary(sess,log_dir)
        
        self.saver = tf.train.Saver(max_to_keep = 20)
        
#         print("-------GLOBAL-------")
#         for w in self.weights_params:
#             print(w)

    def _build_model(self,state):
            self.conv1 = NetworkSetting.conv1(state)
            self.maxpool1 = NetworkSetting.maxpool1(self.conv1)
            self.conv2 = NetworkSetting.conv2(self.maxpool1)
            self.maxpool2 = NetworkSetting.maxpool2(self.conv2)
            reshape = NetworkSetting.reshape(self.maxpool2)
            rnn ,self.rnn_len ,self.row_output = NetworkSetting.lstm(reshape, state)
            fc1 = NetworkSetting.fc1(rnn)

            with tf.variable_scope("policy"):
                self.policy = NetworkSetting.policy(fc1)
            
            with tf.variable_scope("value"):
                self.value = NetworkSetting.value(fc1)
                
            self.policy_prob = tf.nn.softmax(self.policy)
                
            print("---------MODEL SHAPE-------------")
            print(state.get_shape())
            print(self.conv1.get_shape())
            print(self.conv2.get_shape())
            print(reshape.get_shape())
            print(fc1.get_shape())
            print(self.policy.get_shape())
            print(self.value.get_shape())
                
    def _build_summary(self,sess, log_dir):
        
        self.reward_ = tf.placeholder(tf.float32,shape=(), name="reward")
        self.frag_ = tf.placeholder(tf.float32, shape=(), name="frag")
        self.death_ = tf.placeholder(tf.float32, shape=(), name="death")
        self.loss_p_ = tf.placeholder(tf.float32, shape=(), name="loss_policy")
        self.loss_v_ = tf.placeholder(tf.float32, shape=(), name="loss_value")
        self.loss_c_ = tf.placeholder(tf.float32, shape=(), name="loss_class")
        
        with tf.variable_scope("Summary_Score"):
            s = [tf.summary.scalar('reward', self.reward_, family="score"), tf.summary.scalar('frag', self.frag_, family="score"), tf.summary.scalar("death", self.death_, family="score")]
            self.summary_reward = tf.summary.merge(s)
        
        with tf.variable_scope("Summary_Loss"):
            self.summary_loss = tf.summary.merge([tf.summary.scalar('loss_policy', self.loss_p_, family="loss"), tf.summary.scalar('loss_value', self.loss_v_, family="loss"), tf.summary.scalar('loss_class', self.loss_c_, family="loss")])
        
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
        
    def write_score(self,sess, step ,reward, frag, death):
        m = sess.run(self.summary_reward, feed_dict={self.reward_:reward, self.frag_:frag, self.death_:death})
        return self.writer.add_summary(m, step)
    
    def write_loss(self,sess, step, l_p, l_v,l_c):
        m = sess.run(self.summary_loss, feed_dict={self.loss_p_: l_p, self.loss_v_:l_v, self.loss_c_:l_c})
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
        
        self.obs = {}
        self.obs['s1'] = np.zeros((N_ADV, )+ RESOLUTION, dtype=np.float32)
        
        self.states_buff = np.zeros((N_ADV*3-2,) + RESOLUTION, dtype=np.float32)
        self.rewards_buff = np.zeros((N_ADV*3-2,), dtype=np.float32)
        self.actions_buff = np.zeros((N_ADV*3-2, ), dtype=np.float32)
        self.isterminals_buff = np.ones((N_ADV*3-2, ), dtype=np.float32)
        self.buff_pointer = 0
        
    def calc_eps(self, progress):
        if progress < 0.2:
            return EPS_MIN
        elif progress >= 0.2 and progress < 0.8:
            return ((EPS_MAX - EPS_MIN)/ 0.6) * progress + ( EPS_MIN -  (EPS_MAX - EPS_MIN)/ 0.6 * 0.2)
        else :
            return EPS_MAX

    def act_eps_greedy(self, sess, s1, progress):
        assert progress >= 0.0 and progress <=1.0
        
        self.push_obs(s1)
        eps = self.calc_eps(progress)
        if random.random() <= eps:
            p = self.network.get_policy(sess, [self.obs['s1']])[0]
            a_idx = np.random.choice(N_AGENT_ACTION, p=p)
        else:
            a_idx = np.random.randint(N_AGENT_ACTION)
            
        return a_idx
    
    def act_greedy(self, sess, s1):
        p = self.network.get_policy(sess, [self.obs['s1']])[0]
        a_idx = np.random.choice(N_AGENT_ACTION, p=p)
        return a_idx
    
    def get_gradients(self, sess):
        batch={'s1':[], 'actions':[], 'rewards':[], 'isterminals':[]}
        for i in range(N_ADV):
            batch['s1'].append(self.states_buff[i:i+N_ADV])
            batch['actions'].append(self.actions_buff[i+N_ADV])
            batch['isterminals'].append(self.isterminals_buff[i+N_ADV])
            batch['rewards'].append(sum([r*GAMMA**j for j,r in enumerate(self.rewards_buff[i:i+N_ADV])]))
            
        return self.network.get_gradients(sess, batch['s1'], batch['actions'], batch['rewards'], batch['isterminals'])
    
    def train_network(self, sess):
        batch={'s1':[], 'actions':[], 'rewards':[], 'isterminals':[]}
        self.buff_pointer = N_ADV
        for i in range(N_ADV):
            batch['s1'].append(self.states_buff[i:i+N_ADV])
            batch['actions'].append(self.actions_buff[i+N_ADV])
            batch['isterminals'].append(self.isterminals_buff[i+N_ADV])
            batch['rewards'].append(sum([r*GAMMA**j for j,r in enumerate(self.rewards_buff[i:i+N_ADV])]))
            
        self.states_buff = np.roll(self.states_buff, shift=-N_ADV, axis=0)
        self.rewards_buff = np.roll(self.rewards_buff, shift=-N_ADV)
        self.isterminals_buff = np.roll(self.isterminals_buff, shift=-N_ADV)
        self.actions_buff = np.roll(self.actions_buff, shift=-N_ADV)
        self.test_batch = batch
        return self.network.update_parameter_server(sess, batch['s1'], batch['actions'], batch['rewards'], batch['isterminals'])
    
    def push_obs(self, s1):
        self.obs['s1'] = np.roll(self.obs['s1'],shift=-1, axis=0)
        self.obs['s1'][-1] = s1
        
    def clear_obs(self):
        self.obs = {}
        self.obs['s1'] = np.zeros((N_ADV,)+ RESOLUTION, dtype=np.float32)
        
    def push_batch(self, s1, action, reward, isterminal):
        self.states_buff[self.buff_pointer] = s1
        self.actions_buff[self.buff_pointer] = action
        self.rewards_buff[self.buff_pointer] = reward
        self.isterminals_buff[self.buff_pointer] = isterminal
        self.buff_pointer += 1
    
    def clear_batch(self):
        self.states_buff = np.zeros((N_ADV*3-2,) + RESOLUTION, dtype=np.float32)
        self.rewards_buff = np.zeros((N_ADV*3-2,), dtype=np.float32)
        self.actions_buff = np.zeros((N_ADV*3-2, ), dtype=np.float32)
        self.isterminals_buff = np.ones((N_ADV*3-2, ), dtype=np.float32)
        self.buff_pointer = 0
    
    def is_trainable(self):
        if self.buff_pointer == len(self.states_buff):
            return True
        else:
            return False


# In[ ]:


class NetworkLocal(object):
    def __init__(self,name, parameter_server):
        self.name = name
        
        with tf.variable_scope(self.name+"_learner", reuse=tf.AUTO_REUSE):
            self.state1_ = tf.placeholder(tf.float32,shape=(None,N_ADV)+RESOLUTION, name="state_1")
            self.a_ = tf.placeholder(tf.float32, shape=(None,), name="action")
            self.r_adv = tf.placeholder(tf.float32, shape=(None,), name="reward_advantage")
            self.isterminal_ = tf.placeholder(tf.float32, shape=(None,), name="isterminal")
            self.policy, self.value, self.conv1, self.conv2 = self._model(self.state1_)

            self._build_graph()
            
#         self.update_global_weight_params = \
#             parameter_server.optimizer.apply_gradients(zip(self.grads, parameter_server.weights_params_fc))
        self.update_global_weight_params =             parameter_server.optimizer.apply_gradients(zip(self.grads, parameter_server.weights_params))

    def _model(self,state):

#         with tf.variable_scope(self.name + "_nottrainable"):
        conv1 = NetworkSetting.conv1(state)
        maxpool1 = NetworkSetting.maxpool1(conv1)
        conv2 = NetworkSetting.conv2(maxpool1)
        maxpool2 = NetworkSetting.maxpool2(conv2)
        reshape = NetworkSetting.reshape(maxpool2)
        
        with tf.variable_scope(self.name + "_trainable"):
            rnn ,l ,self.row_output = NetworkSetting.lstm(reshape, state)
            fc1 = NetworkSetting.fc1(rnn)

            policy = NetworkSetting.policy(fc1)
            value = NetworkSetting.value(fc1)
            
        self.test_len = l
        self.rnn = rnn
        
        return policy, value, conv1, conv2

    def _build_graph(self):

        one_hot = tf.one_hot(tf.cast(self.a_, tf.int32), depth=N_AGENT_ACTION)
        
        self.policy_prob = tf.nn.softmax(self.policy)
        
        log_prob = -tf.log(tf.reduce_sum(self.policy_prob * one_hot, axis=1)+1e-10)
        advantage = self.r_adv - tf.reshape(self.value,[-1])
        self.loss_policy = log_prob * tf.stop_gradient(advantage)
        self.loss_value = tf.square(advantage) * 0.01
        self.loss_class = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot,logits=self.policy)*tf.stop_gradient(advantage)
#         self.loss_total = tf.reduce_mean(self.loss_policy + self.loss_value)
        self.loss_total = tf.reduce_mean(self.loss_policy + self.loss_value + self.loss_class)
        
        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+"_learner")
        self.weights_params_fc = self.weights_params[4:]
        
#         self.grads = tf.gradients(self.loss_total ,self.weights_params_fc)
        self.grads = tf.gradients(self.loss_total ,self.weights_params)
        
        self.pull_global_weight_params_fc = [l_p.assign(g_p) for l_p,g_p in zip(self.weights_params_fc,parameter_server.weights_params_fc)]
        self.push_local_weight_params_fc = [g_p.assign(l_p) for g_p,l_p in zip(parameter_server.weights_params_fc,self.weights_params_fc)]
        
        self.pull_global_weight_params = [l_p.assign(g_p) for l_p,g_p in zip(self.weights_params,parameter_server.weights_params)]
        self.push_local_weight_params = [g_p.assign(l_p) for g_p,l_p in zip(parameter_server.weights_params,self.weights_params)]
        
    def pull_parameter_server(self, sess):
        sess.run(self.pull_global_weight_params_fc)
    
    def push_parameter_server(self,sess):
        sess.run(self.push_local_weight_params_fc)
        
    def pull_parameter_server_all(self, sess):
        sess.run(self.pull_global_weight_params)
    
    def push_parameter_server_all(self):
        sess.run(self.push_local_weight_params)
        
    def get_weights(self, sess):
        return sess.run(self.weights_params)
    
    def get_gradients(self,sess, s1, a, r, isterminal):
        assert np.ndim(s1) == 5
        
        feed_dict = {self.state1_: s1, self.a_:a, self.r_adv:r,  self.isterminal_:isterminal}
        return sess.run(self.grads, feed_dict)
    
    def update_parameter_server(self,sess, s1, a, r, isterminal):
        assert np.ndim(s1) == 5
        feed_dict = {self.state1_: s1,self.a_:a, self.r_adv:r}
        _, l_p, l_v = sess.run([self.update_global_weight_params, self.loss_policy, self.loss_value],feed_dict)
        return l_p, l_v
    
    def update_parameter_server_imitation(self,sess, s1, a, r, isterminal):
        assert np.ndim(s1) == 5
        feed_dict = {self.state1_: s1,self.a_:a, self.r_adv:r}
        _, l_p, l_v, l_c = sess.run([self.update_global_weight_params, self.loss_policy, self.loss_value, self.loss_class],feed_dict)
        return l_p, l_v, l_c
    
    def check_weights(self, sess):
        weights = SESS.run(self.weights_params)
        assert np.isnan([np.mean(w) for w in weights]).any()==False , print(weights)

    def get_policy(self, sess, s1):
        assert np.ndim(s1) == 5
        return sess.run(self.policy_prob, {self.state1_: s1})

    def get_value(self, sess, s1):
        assert np.ndim(s1) == 5        
        return sess.run(self.value, {self.state1_:s1})


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
        temp = tf.reduce_max(state, axis=4)
        temp = tf.reduce_max(temp, axis=3)
        temp = tf.reduce_max(temp, axis=2)
        lengh = tf.cast(tf.reduce_sum(tf.sign(temp) , axis=1),dtype=tf.int32) 
        cell = tf.nn.rnn_cell.LSTMCell(LSTM_SIZE)
        rnn_state = cell.zero_state(batch_size, dtype=tf.float32)
        rnn_out, state_out = tf.nn.dynamic_rnn(cell, pre_layer, initial_state=rnn_state, dtype=tf.float32)
        out_idx = tf.range(0, batch_size) * N_ADV + (lengh  -1)
        output = tf.gather(tf.reshape(rnn_out, [-1, LSTM_SIZE]), out_idx)
        return output, lengh, rnn_out
    
    def fc1(pre_layer):
        num_outputs =1024
        activation_fn = tf.nn.relu
        weights_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.1)
        return tf.contrib.layers.fully_connected(pre_layer,num_outputs=num_outputs,activation_fn=activation_fn,weights_initializer=weights_init, biases_initializer=bias_init)
    
    def policy(pre_layer):
        num_outputs=N_AGENT_ACTION
        activation_fn = None
        weights_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.1)
        return tf.contrib.layers.fully_connected(pre_layer,num_outputs=num_outputs,activation_fn=activation_fn,weights_initializer=weights_init, biases_initializer=bias_init)
    
    def value(pre_layer):
        num_outputs = 1
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
        game = GameInstance(DoomGame(), name="noname", rewards =REWARDS, config_file_path=CONFIG_FILE_PATH,n_adv=N_ADV)
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
            
            pre_frag = frag[0]
            pre_health = health[0]
            originx = posx[0]
            originy = posy[0]
            pre_ammo = ammo[0]
            rewards = []
            for i in range(1, n_transit):
                r,r_detail = game.get_reward(frag[i]-pre_frag, 0, health[i] - pre_health, ammo[i] - pre_ammo, posx[i] - originx, posy[i]-originy)
                if(i%N_ADV == 0):
                    originx = posx[i]
                    originy = posy[i]
                pre_frag = frag[i]
                pre_health = health[i]
                pre_ammo = ammo[i]
                rewards.append(r)
            rewards = np.array(rewards)
            
            actions = []
            for a_r in action_row:
                ans = 0
                for i, e_a in enumerate(a_r):
                    ans += e_a * 2**i
                actions.append(ans)
            actions = np.array(actions)
            
            for i in range(N_ADV , n_transit - N_ADV - 1):
                s = states_row[i-N_ADV :i]
                a = actions[i]
                r = sum([r_ * GAMMA**j for j,r_ in enumerate(rewards[i:i+N_ADV])])
                batch = [np.copy(s), a, r, False]
                replay_memory.store(batch) 
                
    replay_memory.tree.set_parmanent_data(total_n_transit)
    print(len(replay_memory), "data are stored")


# In[ ]:


replaymemory = ReplayMemory(10000)
rewards = load_demo(replaymemory, DEMO_PATH)


# In[ ]:


if __name__=="learning":
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list=USED_GPU))
    config.log_device_placement = False
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    with tf.device('/gpu:0'):
        TEST_GRADS = []
        parameter_server = ParameterServer(sess,LOGDIR)

        starttime = datetime.now().timestamp()
        end_time = (datetime.now() + timedelta(minutes=10)).timestamp()
        
        coordinator = tf.train.Coordinator()

        name = "worker_imitation"
        game_instance=GameInstance(DoomGame(), name=name, rewards =REWARDS, config_file_path=CONFIG_FILE_PATH,n_adv=N_ADV)
        network = NetworkLocal(name, parameter_server)
        agent = Agent(network)
        imitation_env = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, start_time=starttime, end_time=end_time)
        imitation_env.log_server = parameter_server
        imitation_env.replay_memory = replaymemory
        thread_imitation = threading.Thread(target=imitation_env.run_prelearning, args=(coordinator,))

        name = "test"
        game_instance=GameInstance(DoomGame(), name=name,rewards=REWARDS, config_file_path=CONFIG_FILE_PATH, n_adv=N_ADV)
        network = NetworkLocal(name, parameter_server)
        agent = Agent(network)
        test_env = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, start_time=starttime, end_time=end_time)
        test_env.log_server = parameter_server
        thread_test = threading.Thread(target=test_env.run_test, args=(coordinator,))
        
        parameter_server.write_graph(sess)
        sess.run(tf.global_variables_initializer())
#         parameter_server.load_cnnweights(sess=sess, weights_path=WEIGHTS_PATH)

        print("-----Start IMITATION LEARNING----")
        threads = [thread_imitation, thread_test]
        for t in threads:
            t.start()
        coordinator.join(threads)

        parameter_server.save_model(sess=sess, step=1, model_path=MODEL_PATH)

        GIF_BUFF = []
        REWARD_BUFF = []
        r,f,d,imgs = test_env.test_agent(gif_buff=GIF_BUFF,reward_buff=REWARD_BUFF)
        GIF_BUFF[0].save('gifs/test.gif',save_all=True, append_images=GIF_BUFF[1:], optimize=False, duration=40*4, loop=0)


# In[ ]:


parameter_server.save_model(sess=sess, step=4, model_path="./models/imitation_learn/model_181107/model.ckpt")


# In[ ]:


GIF_BUFF = []
REWARD_BUFF = []
r,f,d,imgs = test_env.test_agent(gif_buff=GIF_BUFF,reward_buff=REWARD_BUFF)
GIF_BUFF[0].save('gifs/test.gif',save_all=True, append_images=GIF_BUFF[1:], optimize=False, duration=40*4, loop=0)


# In[ ]:


if __name__ == "test":
    MODEL_PATH = "./models/copy_params/model_a3c_rnn_copy_181029/model.ckpt"
#     MODEL_PATH = "./models/copy_params/model_a3c_rnn_copy_181105/model.ckpt"
    TEST_GRADS = []
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list=USED_GPU))
    config.log_device_placement = False
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    starttime = datetime.now().timestamp()
    end_time = (datetime.now() + timedelta(minutes=15)).timestamp()
    
    with tf.device('/gpu:0'):
        parameter_server = ParameterServer(sess,LOGDIR)
        parameter_server.load_model(sess=sess, model_path=MODEL_PATH, step=2)
        
        name = "test"
        game_instance=GameInstance(DoomGame(), name=name, config_file_path=CONFIG_FILE_PATH,rewards=REWARDS, n_adv=N_ADV)
        network = NetworkLocal(name, parameter_server)
        agent = Agent(network)
        test_env = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, start_time=starttime, end_time=end_time)


# In[ ]:


test_img_temp = np.load("./test_imgs.npy")
test_img_temp = np.concatenate([test_img_temp, [np.zeros_like(test_img_temp[0])]])

test_img = np.reshape(test_img_temp, (-1, 5)+RESOLUTION)

for i in range(5):
    f = plt.figure()
    ax = f.add_subplot(1,1,1)
    ax.imshow(test_img[10,i])


# In[ ]:


test_img.shape


# In[ ]:


feed_dict = {parameter_server.state1_: test_img[:]}

pol,val,c2 = sess.run([parameter_server.policy_prob,parameter_server.value, parameter_server.conv2], feed_dict)


# In[ ]:


c2 = np.reshape(c2, (10,-1))


# In[ ]:


for i,c in enumerate(c2):
    f = plt.figure()
    ax = f.add_subplot(1,1,1)
    ax.scatter(range(len(c)), c)


# In[ ]:


loss_policy = []
loss_value = []
loss_class = []
for i in range(1000):
    b = replaymemory.tree.data[i]
    feed_dict = {test_env.network.state1_:[b[0]], test_env.network.a_:[b[1]], test_env.network.r_adv:[b[2]]}
    l_p, l_v, l_c = sess.run([test_env.network.loss_policy, test_env.network.loss_value, test_env.network.loss_class], feed_dict)
    loss_policy.append(l_p)
    loss_value.append(l_v)
    loss_class.append(l_c)
loss_policy = np.reshape(np.array(loss_policy), [-1])
loss_value = np.reshape(np.array(loss_value), [-1])
loss_class = np.reshape(np.array(loss_class), [-1])


# In[ ]:


np.where(np.array(loss_policy) > 5000)


# In[ ]:


plt.plot(range(len(loss_policy)), loss_policy)


# In[ ]:


plt.plot(range(len(loss_class)), loss_class)


# In[ ]:


plt.plot(range(len(loss_value)), loss_value)


# In[ ]:


b = replaymemory.tree.data[52]


# In[ ]:


b[2]


# In[ ]:


b[1]


# In[ ]:


for i in range(5):
    f = plt.figure()
    ax = f.add_subplot(1,1,1)
    ax.imshow(b[0][i])


# In[ ]:


plt.plot(range(len(loss_policy)), loss_policy)


# In[ ]:


for p in pol:
    f = plt.figure()
    ax = f.add_subplot(1,1,1)
    ax.bar(range(len(p)),p)


# In[ ]:





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

