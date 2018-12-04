
# coding: utf-8

# In[ ]:


import numpy as np
from vizdoom import *
import skimage.color, skimage.transform
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


LOG_DIR = "./logs/logs_test"
MODEL_PATH = "./models/model_20180918"
PREMODEL_PATH = "./models/premodel_test"
GIF_PATH = "./gifs/test.gif"
PREGIF_PATH = "./gifs/pre_v13.gif"

__name__ = "test"

DEMO_PATH = DEMO_PATH[:3]

TIME_LEARN = datetime.timedelta(minutes = 20)
TIME_PRELEARN = datetime.timedelta(minutes = 1)

SAVE_FILE = True

TOTAL_STEPS = 10000
TOTAL_TIME = TIME_LEARN.seconds
TOTAL_TIME_PRE = TIME_PRELEARN.seconds


# In[ ]:


# --class for Thread　-------
class WorkerThread:
    # Each Thread has an Environment to run Game and Learning.
    def __init__(self, thread_name, parameter_server, replay_memory, isLearning=True):
        self.environment = Environment(thread_name, parameter_server, replay_memory)
        print(thread_name," Initialized")
        self.isLearning = isLearning

    def run(self):
        if self.isLearning:
            while True:
                if not self.environment.finished:
                    self.environment.run()
                else:
                    break
        else:
            # Run Test Environment
            pass


# In[ ]:


class Environment(object):
    def __init__(self,name, parameter_server,replay_memory, summary=False):
        
        self.name = name
        self.game = GameInstance(DoomGame(), name=self.name, config_file_path=CONFIG_FILE_PATH, rewards=REWARDS,n_adv=N_ADV)
        
        self.network = NetworkLocal(name, parameter_server)
        self.agent = Agent(self.network)
        self.replay_memory = replay_memory
        
        self.local_step = 0
        
        self.finished = False
        
        self.summary = summary
        self.parameter_server = parameter_server
        
        self.log_buff = [np.zeros(shape=(N_ADV,) + RESOLUTION, dtype=np.float32),                          np.zeros(shape=(N_ADV,), dtype=np.int8),                          np.zeros(shape=(N_ADV,) +  RESOLUTION, dtype=np.float32),                          np.zeros(shape=(N_ADV,), dtype=np.float32),                          np.ones(shape=(N_ADV,), dtype=np.int8),                          np.zeros(shape=(N_ADV,), dtype=np.int8)]
        self.buff_pointer = 0
        
    def preprocess(self,img):
        if len(img.shape) == 3 and img.shape[0]==3:
            img = img.transpose(1,2,0)
        
        img = skimage.transform.resize(img, RESOLUTION, mode="constant")
        img = img.astype(np.float32)
#         img = (img)/255.0
        return img
    
    def run_pre_learning(self):
        global frames, start_time_pre,current_time
        
        step = 0
        while True:
            
            self.network.pull_parameter_server()
            tree_idx, s1, actions, s2, rewards, rewards_adv, isterminals, isdemos, is_weight, a_onehot = self.make_batch()
#             print(is_weight)
            l_one, l_adv, l_cls, l_l2 = self.network.update_parameter_server_batch(s1,actions,rewards,rewards_adv,s2,isdemos, is_weight, a_onehot, isterminals)

#             print("l_one:{} l_adv:{}, l_cls:{}, l_l2:{}".format(l_one, l_adv, l_cls, l_l2))
            self.replay_memory.batch_update(tree_idx, l_one+l_cls+l_l2, isdemos)
            if step%RECORD_INTERVAL == 0 and SAVE_FILE == True:
#                 lo, la, lc = self.network.calc_loss(s1[0:1],actions[0:1],rewards[0:1],rewards_adv[0:1],s2[0:1],isdemos[0:1])
                self.parameter_server.write_weights(frames)
                self.parameter_server.write_loss(frames, np.mean(l_one), np.mean(l_adv), np.mean(l_cls), l_l2)
                if step % (RECORD_INTERVAL*10) == 0:
                    self.parameter_server.write_images(frames, s1[0:1])
            if step%FREQ_COPY==0:
                self.network.copy_learn2target()
                
            
            if step % TEST_INTERVAL == 0 and SAVE_FILE==True:
                r,frag,death = self.run_test(frames)
                self.parameter_server.write_records(frames, r, frag,death)
            
            step += 1
            frames += 1
            
            current_time = datetime.datetime.now().timestamp() - start_time_pre.timestamp()
            
            if datetime.datetime.now() > TIME_PRELEARN + start_time_pre:
                runout = True
                break

    def run(self):
        global current_time, frames, start_time_async, N_EPISODES
        
        step = 0
        while True:
            self.network.pull_parameter_server()

            if len(self.replay_memory) > BATCH_SIZE:
                tree_idx, s1, actions, s2, rewards, rewards_adv, isterminals, isdemos, is_weight, a_onehot = self.make_batch()
                l_one, l_adv, l_cls, l_l2 = self.network.update_parameter_server_batch(s1,actions,rewards,rewards_adv,s2,isdemos, is_weight, a_onehot, isterminals)
                if step%RECORD_INTERVAL == 0 and SAVE_FILE == True and self.summary == True:
                    self.parameter_server.write_eps(frames, float(self.agent.calc_eps_time()))
                    self.parameter_server.write_weights(frames)
                    self.parameter_server.write_loss(frames, np.mean(l_one), np.mean(l_adv), np.mean(l_cls), l_l2)
                    if step % (RECORD_INTERVAL*10) == 0:
                        self.parameter_server.write_images(frames, s1[0:1])

            s1_ = self.preprocess(self.game.get_screen_buff())
            action = self.agent.act_eps_greedy(s1_)
            r,_ = self.game.make_action(action, FRAME_REPEAT)
#             self.replay_memory.batch_update(tree_idx, l_cls + 0.01, isdemos)

            if step%FREQ_COPY==0:
                self.network.copy_learn2target()
            
            s2_ = self.preprocess(self.game.get_screen_buff()) if not self.game.is_episode_finished() else np.zeros_like(s1_)
            
            self.add_buff(s1_, action.index(1), s2_, r, self.game.is_episode_finished(), False)
            self.buff_pointer += 1
            
            if (self.game.is_player_dead()):
                self.game.respawn_player()
            
            if self.game.is_episode_finished():
                N_EPISODES += 1
                self.game.new_episode(BOTS_NUM)
                while(self.buff_pointer < N_ADV):
                    self.add_buff(np.zeros_like(s1_), -1, np.zeros_like(s1_), 0, True, False)
                    self.buff_pointer += 1
                self.replay_memory.store(self.log_buff)
                self.clear_buff()
                self.agent.clear_memory()
                
            if self.buff_pointer == N_ADV:
                self.replay_memory.store(self.log_buff)
                self.clear_buff()

            step += 1
            frames += 1
            
            current_time = datetime.datetime.now().timestamp() - start_time_async.timestamp()
            
            if runout == True:
                self.finished = True
                break

        return 0
                
    def make_batch(self):
        
        tree_idx, batch, is_weight = self.replay_memory.sample(BATCH_SIZE)
#         tree_idx,batch,is_weight = self.replay_memory.sample_uniform(BATCH_SIZE)

        s1 = np.zeros((BATCH_SIZE , N_ADV,)+RESOLUTION,dtype=np.float32)
        s2 = np.zeros((BATCH_SIZE , N_ADV,)+RESOLUTION,dtype=np.float32)
        actions = np.zeros((BATCH_SIZE ,),dtype=np.int8)
        rewards = np.zeros((BATCH_SIZE, ),dtype=np.float32)
        rewards_adv = np.zeros((BATCH_SIZE,),dtype=np.float32)
        isterminals = np.zeros((BATCH_SIZE, ),dtype=np.int8)
        isdemos = np.zeros((BATCH_SIZE,),dtype=np.int8)
        a_onehot = np.zeros((BATCH_SIZE, N_AGENT_ACTION,), dtype=np.int8)
        
        for i in range(BATCH_SIZE):
            isterminal = (batch[i][4] == 1)
            s1[i] = batch[i][0]
            s2[i] = batch[i][2]
            isterminals[i] = 1 if isterminal.any() else 0
            actions[i] = batch[i][1][isterminal][0] if isterminal.any() else batch[i][1][-1]
            rewards[i] = batch[i][3][isterminal][0] if isterminal.any() else batch[i][3][-1]
            isdemos[i] = batch[i][5][-1]
            rewards_adv[i] = sum([r * GAMMA**j for j,r in zip(range(len(batch[i][3])-1,0,-1), batch[i][3])])
            a_onehot[i][actions[i]] = 1
            
        return tree_idx, s1, actions, s2, rewards, rewards_adv, isterminals, isdemos, is_weight, a_onehot
    
    def run_test(self, global_step, gif_buff=None reward_buff=None, show_result=True):
        
        self.game.new_episode(BOTS_NUM)
        
        #Copy params from global
        self.network.pull_parameter_server()

        step = 0
        gif_img = []
        total_reward = 0
        total_detail = {}
        while not self.game.is_episode_finished():
            s1_row = self.game.get_screen_buff()
            s1 = self.preprocess(s1_row)
            if gif_buff is not None:
                gif_img.append(s1_row.transpose(1,2,0))
            action = self.agent.act_greedy(s1)
            engine_action = self.convert_action_agent2engine(action.index(1))
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
            print(np.shape(gif_img))
            for i in range(len(gif_img)):
                save_img.append(Image.fromarray(np.uint8(gif_img[i])))
            gif_buff += save_img
#             save_img[0].save(GIF_PATH,save_all=True,append_images=save_img[1:])
        if show_result == True:
            print("----------TEST at %d step-------------"%(global_step))
            print("FRAG:",self.game.get_frag_count(), "DEATH:",self.game.get_death_count())
            print("REWARD",total_reward)
            print(total_detail)
        return total_reward, self.game.get_frag_count(), self.game.get_death_count()
    
    def load_demonstration(self):
        
        for d in DEMO_PATH[:]:
            print("Loading ", d)
            file = h5py.File(d,"r")
            episode_list = list(file.keys())[1:]
    #         episode_list = episode_list[:1]
            self.clear_buff()

            for e in episode_list:
                n_steps = file[e+"/states"].shape[0]
                states = file[e+"/states"][:]
                actions = file[e+"/action"][:]
                frags = file[e+"/frag"][:]
                deaths = file[e+"/death"][:]
                health = file[e+"/health"][:]
                ammo = file[e+"/ammo"][:]
                posx = file[e+"/posx"][:]
                posy = file[e+"/posy"][:]

                originx = 0
                originy = 0

                for i in range(n_steps):

                    if i % N_ADV == 0:
                        originx = posx[i]
                        originy = posy[i]

                    if not i == n_steps - 1:

                        m_frag = frags[i+1] - frags[i]
                        m_death = deaths[i+1] - deaths[i]
                        m_health = health[i+1] - health[i]
                        m_ammo = ammo[i+1] - ammo[i]
                        m_posx = abs(posx[i] - originx)
                        m_posy = abs(posy[i] - originy)

                        r,_ = self.game.get_reward(m_frag, m_death, m_health, m_ammo, m_posx, m_posy)

                        s1 = self.preprocess(states[i])
                        s2 = self.preprocess(states[i+1])
                        agent_action_idx = self.convert_action_engine2agent(actions[i].tolist())

                        self.add_buff(s1, agent_action_idx, s2, r, False, True)
                        self.buff_pointer += 1
                    else:

                        r,_ = self.game.get_reward(m_frag,m_death,m_health,m_ammo,m_posx,m_posy) 

                        s1 = self.preprocess(states[i])
                        agent_action_idx = self.convert_action_engine2agent(actions[i].tolist())
                        self.add_buff(s1,agent_action_idx, np.zeros_like(s1), r,True, True)
                        self.buff_pointer += 1
                        while(self.buff_pointer < N_ADV):
                            self.add_buff(np.zeros_like(s1), -1, np.zeros_like(s1), 0, True, True)
                            self.buff_pointer += 1

                    if self.buff_pointer == N_ADV:
                        self.replay_memory.store(self.log_buff)
                        self.clear_buff()
                        
        self.replay_memory.set_n_permanent_data(len(self.replay_memory))
        print("number of demondtration: %d"%(self.replay_memory.permanent_data))
        return 0

    def convert_action_engine2agent(self,engine_action):
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
    
    def add_buff(self, s1,a,s2,r,isterminal, isdemo):
        self.log_buff[0][self.buff_pointer] = s1
        self.log_buff[2][self.buff_pointer] = s2
        self.log_buff[1][self.buff_pointer] = a
        self.log_buff[3][self.buff_pointer] = r
        self.log_buff[4][self.buff_pointer] = isterminal
        self.log_buff[5][self.buff_pointer] = isdemo
        return 0
    
    def clear_buff(self):
        self.log_buff = [np.zeros(shape=(N_ADV,) + RESOLUTION, dtype=np.float32),                          np.zeros(shape=(N_ADV,), dtype=np.int8),                          np.zeros(shape=(N_ADV,) +  RESOLUTION, dtype=np.float32),                          np.zeros(shape=(N_ADV,), dtype=np.float32),                          np.ones(shape=(N_ADV,), dtype=np.int8),                          np.zeros(shape=(N_ADV,), dtype=np.int8)]
        self.buff_pointer = 0
        return 0


# In[ ]:


# Sampling should not execute when the tree is not full !!!
class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity, permanent_data=0):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # stores not probabilities but priorities !!!
        self.data = np.zeros(capacity, dtype=object)  # stores transitions
        self.permanent_data = permanent_data  # numbers of data which never be replaced, for demo data protection
        assert 0 <= self.permanent_data <= self.capacity  # equal is also illegal
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.data_pointer

    def set_n_permanent_data(self,n_permanent_data):
        self.permanent_data = n_permanent_data
        self.data_pointer = self.permanent_data

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.full = True
            self.data_pointer = self.data_pointer % self.capacity + self.permanent_data  # make sure demo data permanent

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]

class ReplayMemory(object):

    epsilon = 0.001  # small amount to avoid zero priority
    demo_epsilon = 1.0  # 1.0  # extra
    alpha = 0.4  # [0~1] convert the importance of TD error to priority
    beta = 0.6  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.00001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity, permanent_data=0):
        self.permanent_data = permanent_data
        self.tree = SumTree(capacity, permanent_data)
#         self.data_name = data_name

    def set_n_permanent_data(self, n):
        self.permanent_data = n
        self.tree.set_n_permanent_data(self.permanent_data)

    def __len__(self):
        return len(self.tree)

    def full(self):
        return self.tree.full

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max_p for new transition
        
    def sample_uniform(self, n):
        num_data = len(self.tree)
        assert num_data > n, print("num_data:{}".format(num_data))
        idx = np.random.randint(0, num_data, (n,))
        b_memory = np.empty((n, ), dtype=object)
        b_memory = self.tree.data[idx]
        ISWeights = np.ones((n,))
        return idx,b_memory,ISWeights
            

    def sample(self, n):
        b_idx = np.empty((n,), dtype=np.int32)
#         b_memory = np.empty((n, self.tree.data[0].size), dtype=object)
        b_memory = np.empty((n,), dtype=object)
        ISWeights = np.empty((n,))
        pri_seg = self.tree.total_p / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        size_replay = len(self.tree)

        if self.tree.full:
            min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
            assert min_prob > 0, "min_prob={}".format(min_prob)

            for i in range(n):
                v = np.random.uniform(pri_seg * i, pri_seg * (i + 1))
                idx, p, data = self.tree.get_leaf(v)  # note: idx is the index in self.tree.tree
                prob = p / self.tree.total_p
                ISWeights[i] = np.power(prob*size_replay, -self.beta)
#                 ISWeights[i] = np.power(prob/min_prob, -self.beta)
#                 ISWeights[i] = prob/min_prob
                b_idx[i], b_memory[i] = idx, data
        else:
            min_prob = np.min(self.tree.tree[self.tree.capacity-1:self.tree.capacity+self.tree.data_pointer-1]) / self.tree.total_p
            assert min_prob > 0, "min_prob={}".format(min_prob)

            for i in range(n):
                if i == 0:
                    v = np.random.uniform(self.abs_err_upper, pri_seg * (i + 1))
                else:
                    v = np.random.uniform(pri_seg * i, pri_seg * (i + 1))
                idx, p, data = self.tree.get_leaf(v)  # note: idx is the index in self.tree.tree
                prob = p / self.tree.total_p
#                 ISWeights[i] = np.power(prob/min_prob, -self.beta)
                ISWeights[i] = np.power(prob*size_replay, -self.beta)
#                 ISWeights[i] = prob/min_prob]
                b_idx[i], b_memory[i] = idx, data

        return b_idx, b_memory, ISWeights  # note: b_idx stores indexes in self.tree.tree, not in self.tree.data !!!

    # update priority
    def batch_update(self, tree_idxes, abs_errors ,is_demo):
        for i, d in enumerate(is_demo):
            if d == True:
                abs_errors[i] += self.demo_epsilon
            else:
                abs_errors[i] += self.epsilon
        
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idxes, ps):
            self.tree.update(ti, p)


# In[ ]:


class Agent(object):
    def __init__(self,q_network):
        
        self.q_network = q_network
        
#         self.image_buff = np.zeros(shape=(N_ADV,)+RESOLUTION)
        self.image_buff = []
        self.memory = []
        self.batch = {'s1':[], 'action':[], 's2':[] ,'reward':[], 'reward_adv':[], 'isdemo':[]}
        self.R = 0
        
        self.s1_record = np.zeros((1,N_ADV,)+RESOLUTION)
        self.loss_one_record = 0
        self.loss_adv_record = 0
        self.loss_class_record = 0
        self.loss_l2_record = 0
        
    def calc_eps_step(self):
            global frames

            if frames<TOTAL_STEPS*LINEAR_EPS_START:
                eps = EPS_START
            elif frames>=TOTAL_STEPS*LINEAR_EPS_START and frames<TOTAL_STEPS*LINEAR_EPS_END:
                eps = EPS_START + frames*(EPS_END-EPS_START)/(TOTAL_STEPS)
            else:
                eps = EPS_END
            return eps
        
    def calc_eps_time(self):
        
        current_time_async = current_time
#         print("%f, %f, %f"%(current_time_async, current_time, TOTAL_TIME_PRE))
        if current_time_async < TOTAL_TIME * LINEAR_EPS_START:
            eps = EPS_START
        elif current_time_async >= TOTAL_TIME * LINEAR_EPS_START and current_time_async < TOTAL_TIME*LINEAR_EPS_END:
            eps = EPS_START + current_time_async*(EPS_END-EPS_START)/(TOTAL_TIME)
        else:
            eps = EPS_END
            
        return eps

    def act_eps_greedy(self,s1):
        
        assert np.ndim(s1) == 3, print("np.ndim(s1)=",np.ndim(s1))

        self.image_buff.append(s1)
        ret_action = np.zeros((N_AGENT_ACTION,))
        
        if not len(self.image_buff) == N_ADV + 1:
            buff = self.image_buff + [np.zeros_like(s1) for _ in range(N_ADV - len(self.image_buff))]
        else:
            self.image_buff.pop(0)
            buff = self.image_buff

        eps = self.calc_eps_time()

#         print("np.shape(buff)",np.shape(buff))
        if random.random() > eps:
            a_idx = self.q_network.predict_best_action(buff)[0]
#             print(a_idx.shape)
        else:
            a_idx = random.randint(0,N_AGENT_ACTION-1)

        ret_action[a_idx] = 1
        return ret_action.tolist()
    
    def act_greedy(self,s1):

        self.image_buff.append(s1)
        ret_action = np.zeros((N_AGENT_ACTION,))
        if len(self.image_buff) == N_ADV + 1:
            self.image_buff.pop(0)
            q = self.q_network.get_q_value(self.image_buff)[0]
            a_idx = self.q_network.predict_best_action(self.image_buff)[0]
        else:
            a_idx = self.q_network.predict_best_action(self.image_buff + [np.zeros(RESOLUTION) for _ in range(N_ADV - len(self.image_buff))])[0]
#             a_idx = randint(0,N_AGENT_ACTION-1)

        ret_action[a_idx] = 1
        return ret_action.tolist()
    
    def push_advantage(self,s1_,a_,r_,s2_,isterminal,isdemo):
        self.memory.append((s1_,a_,r_,s2_,isdemo))
    
    def clear_memory(self):
        self.memory = []
    
    def push_to_batch(self, s1, action, s2, reward, reward_adv, isdemo):
        self.batch['s1'].append(s1)
        self.batch['action'].append(action)
        self.batch['s2'].append(s2)
        self.batch['reward'].append(reward)
        self.batch['reward_adv'].append(reward_adv)
        self.batch['isdemo'].append(isdemo)
        return 0
    
    def clear_batch(self):
        self.batch = {'s1':[], 'action':[], 's2':[] ,'reward':[], 'reward_adv':[], 'isdemo':[]}
        return 0
    
    def make_batch_learn(self):
        n = len(self.batch['action'])
        s1 = np.zeros((n, N_ADV,)+RESOLUTION)
        s2 = np.zeros((n, N_ADV,)+RESOLUTION)
        for i in range(n):
            s1[i, :n - i] = self.batch['s1'][:n - i]
            s2[i, :n - i] = self.batch['s2'][:n - i]
        
        self.s1_record = s1[0:1]
        
        self.loss_one_record, self.loss_adv_record, self.loss_class_record =         self.q_network.update_parameter_server_batch(s1, self.batch['action'], self.batch['reward'],                                                          self.batch['reward_adv'], s2, self.batch['isdemo'])
        return 0
    
    def get_q_value(self):
        
        if len(self.image_buff) == N_ADV:
            q = self.q_network.get_q_value(self.image_buff)[0]
        else:
            q = self.q_network.get_q_value(self.image_buff + [np.zeros(RESOLUTION) for _ in range(N_ADV - len(self.image_buff))])[0]
        
        return q
        
    
    def learn_advantage(self, isterminal):
        
        if len(self.memory)==N_ADV or isterminal:
            tail_idx = len(self.memory)-1
            
            s1_buff = np.zeros((N_ADV, )+RESOLUTION)
            for i in range(tail_idx+1):
                s1_buff[i] = self.memory[i][0]
            
            for i in range(tail_idx,-1,-1):
                s1,a,r,s2,d = self.memory[i]
                if i==tail_idx:
                    if not isterminal:
#                         print(np.max(self.q_network.get_q_value(s1)[0]))
                        self.R = np.max(self.q_network.get_q_value(s1_buff)[0])
                        
                    else:
                        self.R = 0
                else:
                    self.R =  r + GAMMA*self.R
            
#                 self.q_network.train_push(s1,a,r,self.R,s2,d)
                self.push_to_batch(s1,a,s2,r,self.R,d)
            
#             self.q_network.update_parameter_server()
#             self.q_network.update_parameter_server_batch(self.batch['s1'], self.batch['action'], self.batch['reward'], \
#                                                          self.batch['reward_adv'], self.batch['s2'], self.batch['isdemo'])

#             print(np.shape(self.batch['s1']))
#             print(np.shape(self.batch['s2']))
#             print(np.shape(self.batch['s2']))
            self.make_batch_learn()
            self.q_network.copy_learn2target()
            self.R = 0
            self.clear_memory()
            self.clear_batch()
            
#             return self.q_network.calc_loss([s1],[a],[r],[self.R],[s2],[d])
#         return 0.0,0.0,0.0
    
    def calc_loss(self):
        
        if len(self.memory) == N_ADV :
            tail_idx = len(self.memory) - 1
            s1_buff = np.ones((1, tail_idx+1, )+RESOLUTION) * np.nan
            s2_buff = np.ones((1, tail_idx+1, )+RESOLUTION) * np.nan
            for i in range(tail_idx+1):
                s1_buff[0, i] = self.memory[i][0]
                s2_buff[0, i] = self.memory[i][3]
            
            for i in range(tail_idx, -1, -1):
                s1 , a, r, s2, d = self.memory[i]
                if i == tail_idx :
                    R = np.max(self.q_network.get_q_value(s1_buff)[0])
                else:
                    R = r * GAMMA * R
                
                _, last_action, last_r, _, last_d = self.memory[tail_idx]
                
                return [s1_buff] + self.q_network.calc_loss(s1_buff, [last_action], [last_r], [R] ,s2_buff ,[last_d])
        
        return -1


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
    
    def fc1(pre_layer):
        num_outputs = 512
        activation_fn = tf.nn.relu
        weights_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.1)
        return tf.contrib.layers.fully_connected(pre_layer,num_outputs=num_outputs,activation_fn=activation_fn,                                                 weights_initializer=weights_init, biases_initializer=bias_init)
    
    def q_value(pre_layer):
        num_outputs = N_AGENT_ACTION
        activation_fn = None
        weights_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.1)
        return tf.contrib.layers.fully_connected(pre_layer,num_outputs=num_outputs,activation_fn=activation_fn,                                                 weights_initializer=weights_init, biases_initializer=bias_init)


# In[ ]:


# Network which be shared in global
class ParameterServer:
    def __init__(self):
        
        self.state1_ = tf.placeholder(tf.float32,shape=(None,N_ADV)+RESOLUTION, name="state1")
        self.a_ = tf.placeholder(tf.int32, shape=(None,), name="action")
        self.r_ = tf.placeholder(tf.float32, shape=(None,), name="reward")
        self.r_adv = tf.placeholder(tf.float32, shape=(None,), name="reward_adv")
        self.mergin_value = tf.placeholder(tf.float32,shape=(None,N_AGENT_ACTION), name="mergin_value")
#         self.s1idx_ = tf.placeholder(tf.int32, shape=(None,), name="lengh_of_state")
        
        with tf.variable_scope("parameter_server",reuse=tf.AUTO_REUSE):      # スレッド名で重み変数に名前を与え、識別します（Name Space）
            self.model = self._build_model()            # ニューラルネットワークの形を決定
            
        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="parameter_server")
#         self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, RMSProbDecaly)    # loss関数を最小化していくoptimizerの定義です
        self.optimizer = tf.train.AdamOptimizer()
        with tf.device("/cpu:0"):
            with tf.variable_scope("summary"):
                self._build_summary()

            self.saver = tf.train.Saver()
        
        print("-------GLOBAL-------")
        for w in self.weights_params:
            print(w)

    def _build_model(self):
        self.conv1 = NetworkSetting.conv1(self.state1_)
        self.maxpool1 = NetworkSetting.maxpool1(self.conv1)
        self.conv2 = NetworkSetting.conv2(self.maxpool1)
        self.maxpool2 = NetworkSetting.maxpool2(self.conv2)
        self.reshape = NetworkSetting.reshape(self.maxpool2)
        self.rnn, self.length, self.rnn_raw = NetworkSetting.lstm(self.reshape, self.state1_)
        self.fc1 = NetworkSetting.fc1(self.rnn)

        q_value = NetworkSetting.q_value(self.fc1)

        print("---------MODEL SHAPE-------------")
        print(self.state1_.get_shape())
        print(self.conv1.get_shape())
        print(self.conv2.get_shape())
        print(self.reshape.get_shape())
        print(self.fc1.get_shape())
        print(q_value.get_shape())

        return q_value

    def _build_summary(self):
        
        self.loss_one = tf.placeholder(tf.float32,shape=())
        self.loss_n = tf.placeholder(tf.float32,shape=())
        self.loss_c = tf.placeholder(tf.float32,shape=())
        self.loss_l = tf.placeholder(tf.float32,shape=())
        
        self.reward = tf.placeholder(tf.float32,shape=())
        self.frag = tf.placeholder(tf.int64,shape=())
        self.death = tf.placeholder(tf.int64,shape=())
        
        summary_lo = tf.summary.scalar('loss_one',self.loss_one, family='loss')
        summary_ln = tf.summary.scalar('loss_nstep', self.loss_n, family='loss')
        summary_lc = tf.summary.scalar('loss_class', self.loss_c, family='loss')
        summary_ll = tf.summary.scalar('loss_l2',self.loss_l, family='loss')

        self.merged_loss = tf.summary.merge([summary_lo,summary_ln,summary_lc,summary_ll])
        
        conv1_display = tf.expand_dims(tf.transpose(self.conv1, perm=[0,1,4,2,3]), axis=5)
        conv2_display = tf.expand_dims(tf.transpose(self.conv2, perm=[0,1,4,2,3]), axis=5)

        state_shape = self.state1_.get_shape()
        conv1_shape = conv1_display.get_shape()
        conv2_shape = conv2_display.get_shape()
        print("state1_shape",state_shape)
        print("conv1_shape:", conv1_shape)
        print("conv2_shape:",conv2_shape)
        summary_state  = tf.summary.image('state',tf.reshape(self.state1_,[-1,state_shape[2], state_shape[3], state_shape[4]]),max_outputs = 1)
        summary_conv1 = tf.summary.image('conv1',tf.reshape(conv1_display,[-1, conv1_shape[3], conv1_shape[4], conv1_shape[5]]),max_outputs = 1)
        summary_conv2 = tf.summary.image('conv2',tf.reshape(conv2_display,[-1, conv2_shape[3], conv2_shape[4], conv2_shape[5]]),max_outputs = 1)

        self.merged_image = tf.summary.merge([summary_state,summary_conv1,summary_conv2])
        
        summary_reward = tf.summary.scalar('reward',self.reward)
        summary_frag = tf.summary.scalar('frag',self.frag)
        summary_death = tf.summary.scalar('death',self.death)
        
        self.merged_testscore = tf.summary.merge([summary_reward,summary_frag,summary_death])
        
        self.merged_weights = tf.summary.merge([tf.summary.scalar(self.weights_params[i].name,tf.reduce_mean(self.weights_params[i]), family='weights') for i in range(len(self.weights_params))])
        
        self.eps_ = tf.placeholder(tf.float32, shape=())
        summary_eps = tf.summary.scalar('epsilon', self.eps_, family='epsilon')
        self.merged_eps = tf.summary.merge([summary_eps])
        
        self.writer = tf.summary.FileWriter(LOG_DIR,SESS.graph)

    # write summary about LOSS and IMAGE
    def write_loss(self,step,loss_one,loss_n,loss_class,loss_l2):
            m = SESS.run(self.merged_loss,feed_dict=                                {self.loss_one:loss_one,self.loss_n:loss_n,self.loss_c:loss_class,self.loss_l:loss_l2})
            self.writer.add_summary(m, step)
            return 0
                
    def write_images(self, step, s1):
        m = SESS.run(self.merged_image, {self.state1_: s1})
        self.writer.add_summary(m, step)
        return 0
    
    def write_records(self,step,r,f,d):
        m = SESS.run(self.merged_testscore,feed_dict={self.reward:r,self.frag:f,self.death:d})
        self.writer.add_summary(m,step)
        
    def write_weights(self, step):
        m = SESS.run(self.merged_weights)
        self.writer.add_summary(m, step)
        return 0
    
    def write_eps(self, step, eps):
        m = SESS.run(self.merged_eps, feed_dict={self.eps_:eps})
        self.writer.add_summary(m, step)
        return 0
    
    def save_model(self, model_path):
        self.saver.save(SESS, model_path+"/model.ckpt")
        
    def load_model(self, model_path):
        self.saver.restore(SESS, model_path+"/model.ckpt")


# In[ ]:


class NetworkLocal(object):
    def __init__(self,name,parameter_server):
        self.name = name
        
        self.state1_ = tf.placeholder(tf.float32,shape=(None,N_ADV,)+RESOLUTION, name="state_1")
        self.state2_ = tf.placeholder(tf.float32,shape=(None,N_ADV,)+RESOLUTION, name="state_2")
        self.a_ = tf.placeholder(tf.int32, shape=(None,), name="action")
        self.a_onehot_ = tf.placeholder(tf.int32, shape=(None, N_AGENT_ACTION,), name="action_onehot")
        self.r_ = tf.placeholder(tf.float32, shape=(None,), name="rewrad")
        self.r_adv = tf.placeholder(tf.float32, shape=(None,), name="reward_advantage")
        self.isdemo_ = tf.placeholder(tf.float32,shape=(None,), name="isdemo")
        self.isterminal_ = tf.placeholder(tf.float32, shape=(None,), name="isterminal")
        self.mergin_value = tf.placeholder(tf.float32,shape=(None,N_AGENT_ACTION), name="mergin")
        self.is_weight_ = tf.placeholder(tf.float32, shape=(None,), name="is_weight")
        self.loss_weights_ = tf.placeholder(tf.float32, shape =(4,), name="loss_weights")
        
        with tf.variable_scope(self.name+"_train", reuse=tf.AUTO_REUSE):
            self.model_l, self.len_s1 = self._model(self.state1_)
        with tf.variable_scope(self.name+"_target", reuse=tf.AUTO_REUSE):
            self.model_t, self.len_s2 = self._model(self.state2_)

        self._build_graph(parameter_server)

#         print("-----LOCAL weights---")
#         for w in self.weights_params:
#             print(w)
            
#         print("-----LOCAL grads---")
#         for w in self.grads:
#             print(w)
    
    def _model(self,state):

        conv1 = NetworkSetting.conv1(state)
        maxpool1 = NetworkSetting.maxpool1(conv1)
        conv2 = NetworkSetting.conv2(maxpool1)
        maxpool2 = NetworkSetting.maxpool2(conv2)
        reshape = NetworkSetting.reshape(maxpool2)
        rnn ,l ,_ = NetworkSetting.lstm(reshape, state)
        fc1 = NetworkSetting.fc1(rnn)
        
        q_value = NetworkSetting.q_value(fc1)
        
        return q_value, 0

    def _build_graph(self,parameter_server):
        
#         self.best_action = tf.argmax(self.model_l, axis=1)
        self.prob_action = tf.nn.softmax(self.model_l, axis=1)
        
#         q_model_t = tf.where(tf.equal(self.len_s2, self.len_s1) , self.model_t,tf.zeros_like(self.model_t))
        q_model_t = tf.where(tf.equal(self.isterminal_, tf.zeros_like(self.isterminal_)) ,self.model_t,tf.zeros_like(self.model_t))
        self.q_model_t = q_model_t
        
        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+"_train")
        self.weights_params_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+"_target")
        self.copy_params = [t.assign(l) for l,t in zip(self.weights_params, self.weights_params_target)]
        
#         self.loss_one = tf.square(tf.stop_gradient(self.r_ + tf.reduce_max(q_model_t,axis=1)) - tf.reduce_max(self.model_l,axis=1))
#         self.loss_adv = tf.square(tf.stop_gradient(self.r_adv + tf.reduce_max(q_model_t,axis=1)) - tf.reduce_max(self.model_l,axis=1))
        self.loss_one = (tf.square(tf.stop_gradient(self.r_ + tf.reduce_max(q_model_t,axis=1)) - tf.reduce_max(self.model_l,axis=1)))
        self.loss_adv = self.loss_weights_[1] * (tf.square(tf.stop_gradient(self.r_adv + np.power(GAMMA, N_ADV) * tf.reduce_max(q_model_t,axis=1)) -                                              tf.reduce_max(self.model_l,axis=1)))
        target = tf.stop_gradient(tf.reduce_max(self.model_l + self.mergin_value, axis=1))
        idx = tf.transpose([tf.range(tf.shape(self.model_l)[0]), self.a_])
        self.loss_class =  self.loss_weights_[2] * ((target- tf.gather_nd(self.model_l,indices=idx)) * self.isdemo_)
        self.loss_l2 = self.loss_weights_[3] * tf.reduce_sum([tf.nn.l2_loss(w) for w in self.weights_params])
        
        self.target_test = target
        self.idx_test = idx
        self.gather_test = tf.gather_nd(self.model_l,indices=idx)
        
        self.loss_total = (tf.reduce_mean(self.loss_adv) +  tf.reduce_mean(self.loss_class) + self.loss_l2) * self.is_weight_
#         self.loss_total = tf.reduce_mean(self.loss_class) + self.loss_l2
        
        self.grads = tf.gradients(self.loss_total ,self.weights_params)
        
        self.update_global_weight_params =             parameter_server.optimizer.apply_gradients(zip(self.grads, parameter_server.weights_params))

        self.pull_global_weight_params = [l_p.assign(g_p) for l_p,g_p in zip(self.weights_params,parameter_server.weights_params)]

        self.push_local_weight_params = [g_p.assign(l_p) for g_p,l_p in zip(parameter_server.weights_params,self.weights_params)]
        
    def test(self, s1, a, r, r_adv, s2, isdemo,isterminal):
        mergin = [[MERGIN_BASE*(not(a[j]==i)) for i in range(N_AGENT_ACTION)] for j in range(len(a))]
        
        feed_dict = {self.state1_: s1,self.a_:a, self.r_:r,self.r_adv:r_adv, self.state2_:s2, self.mergin_value:mergin,self.isdemo_:isdemo, self.isterminal_:isterminal}
        print(SESS.run([self.q_model_t, self.target_test, self.idx_test, self.gather_test], feed_dict))
        return 0
    
    def pull_parameter_server(self):
        SESS.run(self.pull_global_weight_params)
    
    def push_parameter_server(self):
        SESS.run(self.push_local_weight_params)
        
    def show_weights(self):
        hoge = SESS.run(self.weights_params)
        for i in range(len(hoge)):
            print(hoge[i])
    
    def update_parameter_server_batch(self, s1, a, r, r_adv, s2, isdemo, is_weight, a_onehot, isterminal):
        if np.ndim(s1) == 4:
            s1 = np.array([s1])            
            
        if np.ndim(s2) == 4:
            s2 = np.array([s2])
        mergin = [[MERGIN_BASE*(not(a[j]==i)) for i in range(N_AGENT_ACTION)] for j in range(np.shape(a)[0])]
        
        if np.shape(s1) != (BATCH_SIZE, N_ADV)+RESOLUTION:
            print(np.shape(s1))
            return 0, 0, 0, 0
        weights = SESS.run(self.weights_params)
        assert np.isnan([np.mean(w) for w in weights]).any()==False , print(weights)
#         print("s1.shape:",s1.shape, "np.mean(s1)", np.mean(s1))
#         print("s2.shape:",s2.shape, "np.mean(s2)", np.mean(s2))
#         print("action:", a)
#         print("reward:", r)
#         print("reward_adv", r_adv)

        feed_dict = {self.state1_: s1,self.a_:a, self.r_:r,self.r_adv:r_adv, self.state2_:s2, self.mergin_value:mergin,self.isdemo_:isdemo,                      self.is_weight_: is_weight, self.a_onehot_:a_onehot, self.isterminal_:isterminal,self.loss_weights_:[1.0, LAMBDA1, LAMBDA2, LAMBDA3]}
        val = SESS.run([self.loss_one, self.loss_adv, self.loss_class, self.loss_l2, self.model_l,self.q_model_t,  self.grads, self.model_l],feed_dict)
#         val = SESS.run([self.update_global_weight_params,self.loss_one, self.loss_adv, self.loss_class, self.loss_l2],feed_dict)
        SESS.run([self.update_global_weight_params],feed_dict)
    
#         TEST_VALUES.append(val)
        
#         print("{}+{}={}".format(val[2],val[3],val[2] + val[3]))
#         print("model_l",val[4])
#         print("model_t", val[5])
#         print("max(Q(s_t) + l(a, a_E)) = ", val[6])
#         print("a_E = ", val[7])
#         print("Q(s_t)[a_E] = ", val[8])
#         print("l_one:{} l_adv:{}, l_cls:{}, l_l2:{}".format(val[0], val[1], val[2], val[3]))
#         print("np.mean(grads)=", [np.mean(g) for g in val[9]])
        
        return val[0], val[1], val[2], val[3]

    def predict_best_action(self, s1):
        if np.ndim(s1)==4:
            s1 = np.array([s1])
        
#         print(np.shape(s1))
#         print(SESS.run(self.model_l, {self.state1_ : s1}))
#         return SESS.run(self.best_action,{self.state1_ : s1})

        probs = SESS.run(self.prob_action, {self.state1_:s1})
        return [np.random.choice(N_AGENT_ACTION, p=p) for p in probs]

    def get_q_value(self,s1):
        if np.ndim(s1)==4:
            s1 = np.array([s1])
            
        return SESS.run(self.model_l,{self.state1_:s1})
    
    def calc_loss(self, s1, a, r, r_adv, s2, isdemo,isterminal):
        mergin = [[MERGIN_BASE*(not(a[j]==i)) for i in range(N_AGENT_ACTION)] for j in range(len(a))]
        
        feed_dict = {self.state1_: s1,self.a_:a, self.r_:r,self.r_adv:r_adv, self.state2_:s2,                      self.mergin_value:mergin,self.isdemo_:isdemo, self.isterminal_:isterminal, self.loss_weights_:[1.0, LAMBDA1, LAMBDA2, LAMBDA3]}
        return SESS.run([self.loss_one, self.loss_adv, self.loss_class, self.loss_l2],feed_dict)
    
    def copy_learn2target(self):
        SESS.run(self.copy_params)

    def train_push(self,s1,a,r,r_adv,s2,isdemo):
        # Push obs to make batch
        self.s1[self.queue_pointer] = s1
        self.s2[self.queue_pointer] = s2
        self.action[self.queue_pointer] = a
        self.reward[self.queue_pointer] = r
        self.reward_adv[self.queue_pointer] = r_adv
        self.isdemo[self.queue_pointer] = isdemo
        self.queue_pointer += 1


# In[ ]:


# class TensorRecorder(object):
#     def __init__(self):
#         self.build_qvalue()
        
#     def build_qvalue(self):
#         self.q_value = tf.placeholder(tf.float32,shape=(N_AGENT_ACTION,), name="q_value")
#         summary_q = tf.summary.histogram('q_value',self.q_value, family='q_value')
#         self.merge_q = tf.summary.merge([summary_q])
        
#     def write_qvalue(self,step, q_value, writer):
#         m = SESS.run(self.merge_q, {self.q_value: q_value})
#         writer.add_summary(m, step)
#         return 0
    
#     def build_score(self):
#         self.reward = tf.placeholder(tf.float32,shape=(), name="reward")
#         summary_reward = tf.summary.scalar('reward',self.reward)
#         self.merged_testscore = tf.summary.merge([summary_reward])
    
#     def write_score(self,step,reward, writer):
#         m = SESS.run(self.merged_testscore,{self.reward:reward})
#         writer.add_summary(m,step)
#         return 0
    


# In[ ]:


if __name__=="learning":
#     recorder = TensorRecorder()
    frames = 0
    runout = False
    current_time = 0
    start_time_async = 0
    
    GIF_BUFF = []
    N_EPISODES = 0

    config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list=USED_GPU))
    config.log_device_placement = False
    config.allow_soft_placement = True
    SESS = tf.Session(config=config)
    
    replay_memory = ReplayMemory(20000,permanent_data=0)

    threads = []
    with tf.device("/gpu:0"):
        parameter_server = ParameterServer()

        for i in range(N_WORKERS):
            threads.append(WorkerThread("learning_"+str(i),parameter_server, replay_memory))

    with tf.device("/gpu:0"):
        pre_env = Environment("pre_env",parameter_server, replay_memory)
        test_env = Environment("test_env", parameter_server,replay_memory)

    SESS.run(tf.global_variables_initializer())

    threads[0].environment.summary=True

    print("---LOADING DEMO---")
    pre_env.load_demonstration()
    print("---PRE LEARNING---")
    start_time_pre = datetime.datetime.now()
    pre_env.run_pre_learning()
    if SAVE_FILE == True:
        print("---SAVING_MODEL---")
        parameter_server.save_model(PREMODEL_PATH)

    print("---ASYNC_LEARNING---")
    start_time_async = datetime.datetime.now()
    current_time = 0
    for worker in threads:
        job = lambda: worker.run()
        t = threading.Thread(target=job)
        t.start()

    test_frame = 0
    while True:
        if frames >= test_frame and frames<test_frame+TEST_INTERVAL:
            parameter_server.write_weights(frames)
            r,frag,death = test_env.run_test(frames)
            if SAVE_FILE == True:
                parameter_server.write_records(frames,r,frag,death)
            test_frame += TEST_INTERVAL
        elif frames >= test_frame+TEST_INTERVAL:
            print("TEST at %d~%d step cant be finished"%(test_frame, test_frame+TEST_INTERVAL-1))
            test_frame += TEST_INTERVAL
        else:
            pass

        if datetime.datetime.now() > TIME_LEARN + start_time_async:
            runout = True
            break
    
    if (SAVE_FILE==True):
        test_env.run_test(global_step = frames, gif_buff=GIF_BUFF)
        GIF_BUFF[0].save(GIF_PATH,save_all=True,append_images=GIF_BUFF[1:])
    
        parameter_server.save_model(MODEL_PATH) 
        
    print(N_EPISODES, "episodes is passed")


# In[ ]:


if __name__=='test':
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list=USED_GPU))
    config.log_device_placement = False
    config.allow_soft_placement = True
    SESS = tf.Session(config=config)
    
    replay_memory = ReplayMemory(CAPACITY,permanent_data=0)
    
    with tf.device("/cpu:0"):
        parameter_server = ParameterServer()
        pre_env = Environment("pre_env",parameter_server, replay_memory)
        
    SESS.run(tf.global_variables_initializer())
    parameter_server.load_model(PREMODEL_PATH)
    pre_env.load_demonstration()
    pre_env.network.pull_parameter_server()


# In[ ]:


if __name__ == "test":
    def show_command(action):
        buff = ""
        action_names = ["TURN_LEFT","TURN_RIGHT","MOVE_RIGHT","MOVE_FORWARD","MOVE_LEFT","ATTACK"]
        if (type(action) == type(list())):
            for i,a in enumerate(action):
                if a==1:
                    buff += action_names[i] + ","
        else:
            for i in range(6):
                if action %2 == 1:
                    buff +=action_names[i] + ","
                action = int(action/2)
        
        return buff + "\n"
    
    def get_q_values(data, env):
        n_data = data.shape[0]
        ans = []
        for i in range(n_data):
            d = data[i]
            q_s =  env.network.get_q_value(d)[0]
            ans.append(q_s)

        return np.array(ans)

    def predict_action(data, env):
        n_data = data.shape[0]
        ans = []
        for i in range(n_data):
            d = data[i]
            q_s =  env.network.get_q_value(d)[0]
            ans.append(np.argmax(q_s))

        return np.array(ans)
    
    def softmax(a):
        c = np.max(a)

        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)

        y = exp_a / sum_exp_a

        return y 


# In[ ]:


if __name__=="test":
    pre_env.network.copy_learn2target()
    lengh = len(replay_memory)
    data = [replay_memory.tree.data[i][0] for i in range(lengh)]
    target_action = np.array([replay_memory.tree.data[i][1][-1] for i in range(lengh)])

    q_values = get_q_values(np.array(data)[:200], pre_env)
    predicted_action = predict_action(np.array(data)[:200], pre_env)
    print(target_action[:200] == predicted_action)

