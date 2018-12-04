
# coding: utf-8

# In[1]:


import numpy as np
from vizdoom import *
import skimage.color, skimage.transform
from random import sample, randint, random
from tqdm import tqdm
import transition
import tensorflow as tf
import replay_memory
import transition
import h5py
import math


# In[2]:


DEMO_PATH = "./demonstration/demodata_cig2017_v0-2.h5"
CONFIG_FILE_PATH = "./config/custom_config.cfg"
LOG_DIR = "./logs_v01"

RESOLUTION = (120,180,3)

N_ADV = 10

FREQ_COPY = 10
FREQ_TEST = 50

N_PRESTEPS = 5
N_STEPS = 200
TOTAL_STEPS = N_STEPS

DISCOUNT = 0.9
LEARNING_RATE = 0.5
FRAME_REPEAT = 10
BATCH_SIZE = 64
LAMBDA1 = 1.0
LAMBDA2 = 1.0
LAMBDA3 = 10e-5
L_MIN = 0.8

N_ACTION = 6

BOTS_NUM = 20

N_FOLDER = 2

REWARDS = {'living':-0.01, 'health_loss':-1, 'medkit':50, 'ammo':0.0, 'frag':500, 'dist':3e-2, 'suicide':-500} 

CAPACITY = 10000

EPS_START = 0.5
EPS_END = 0.0
LINEAR_EPS_START = 0.1
LINEAR_EPS_END = 0.9


# In[3]:


class Environment(object):
    def __init__(self,name):
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
        
        self.replay_buff = replay_memory.ReplayMemory(CAPACITY,data_name="demodata_cig2017.npy")
        self.network = Network()
        self.agent = Agent(self.network,self.replay_buff,self.reward_gen)
        
        self.local_step = 0
        
        self.finished = False
        
        self.name = name

    
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
    
        return r, r_detail
    
    def load_demonstration(self,isnpy=False):
        if isnpy:
            self.replay_buff.load_data(dir_path)
        else:
            hdf5file = h5py.File(DEMO_PATH,"r")
            folder = "demodata_"+str(0)
            state1 = hdf5file[folder+"/state1"].value
            state2 = hdf5file[folder+"/state2"].value
            actions = hdf5file[folder+"/actions"].value
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
                actions = np.concatenate((actions,hdf5file[folder+"/actions"].value),axis=0)
                isterminals = np.concatenate((isterminals,hdf5file[folder+"/isterminals"].value),axis=0)
                health = np.concatenate((health,hdf5file[folder+"/healths"].value),axis=0)
                ammo = np.concatenate((ammo,hdf5file[folder+"/ammos"].value),axis=0)
                posx = np.concatenate((posx,hdf5file[folder+"/posxs"].value),axis=0)
                posy = np.concatenate((posy,hdf5file[folder+"/posys"].value),axis=0)
                death = np.concatenate((death,hdf5file[folder+"/deaths"].value),axis=0)
                frag = np.concatenate((frag,hdf5file[folder+"/frags"].value),axis=0)

            n_transit, n_step, _ = actions.shape
            
            print("SIZE of DEMO:",actions.shape)

            transit = np.empty((n_step,),dtype=object)

            is_dead = False
            is_finished = False

            pre_health = 100
            pre_ammo = 15
            pre_frag = 0
            pre_death = 0
            pre_posx = 0.0
            pre_posy = 0.0


            for i in range(n_transit):

                if i % 2 == 0:
                    pre_posx = posx[i][0]
                    pre_posy = posy[i][0]

                for j in range(n_step):
                    if not is_finished:
                        if is_dead :
                            pre_posx = posx[i][j]
                            pre_posy = posy[i][j]
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
                        transit[j] = transition.Transition(state1[i][j],actions[i][j],state2[i][j],r,isterminals[i][j],True)

                        pre_frag = frag[i][j]
                        pre_death = death[i][j]
                        pre_health = health[i][j]
                        pre_ammo = ammo[i][j]
                    else:
                        transit[j] = transition.Transition(None,None,None,None,True,True)
                        
                is_finished = False
                
                self.replay_buff.store(np.copy(transit))
                
    def test_score(self):        

        pre_frag = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        pre_death = self.game.get_game_variable(GameVariable.DEATHCOUNT)

        self.start_episode()
        
        total_reward = 0
        total_reward_detail = {'living':0.0, 'health_loss':0.0, 'medkit':0.0, 'ammo':0.0, 'frag':0.0, 'dist':0.0, 'suicide': 0.0}
        
        total_steps = 0
        
        posx = self.game.get_game_variable(GameVariable.POSITION_X)
        posy = self.game.get_game_variable(GameVariable.POSITION_Y)
        self.reward_gen.new_episode(100,15,posx,posy)

        while not self.game.is_episode_finished():
            if self.game.is_player_dead():
                self.game.respawn_player()
                
                posx = self.game.get_game_variable(GameVariable.POSITION_X)
                posy = self.game.get_game_variable(GameVariable.POSITION_Y)
                self.reward_gen.respawn_pos(100,15,posx,posy)
                
            if total_steps%2 == 0:
                self.reward_gen.update_origin(self.game.get_game_variable(GameVariable.POSITION_X),                                                  self.game.get_game_variable(GameVariable.POSITION_Y))

            if self.game.is_episode_finished():
                break

            action = self.agent.act_greedy(self.preprocess(self.game.get_state().screen_buffer))

            self.game.make_action(action, 5)

            reward, r_d  = self.get_reward()
            total_reward += reward
            for k,v in r_d.items():
                total_reward_detail[k] += v
            
            total_steps += 1
            

        frag_count = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        death_count = self.game.get_game_variable(GameVariable.DEATHCOUNT)

        return total_steps,total_reward, total_reward_detail,frag_count-pre_frag, death_count-pre_death
                
    def pre_learning(self):
        
        global frames
        for i in tqdm(range(N_PRESTEPS)):
            self.agent.perform_pre_learning_step()
            frames += 1
    
    def run(self):
#         global frames,runout
        global frames
    
        self.start_episode()
        
        train_episode = 0
        step = 0
        transitions = np.empty((N_ADV,),dtype=object)
        for step in tqdm(range(N_STEPS)):

            if not self.game.is_episode_finished():
                
                if step%N_ADV==0 and not step==0:
                    self.replay_buff.store(np.copy(transitions))
                    self.reward_gen.update_origin(self.game.get_game_variable(GameVariable.POSITION_X),                                                  self.game.get_game_variable(GameVariable.POSITION_Y))

                s1 = self.preprocess(self.game.get_state().screen_buffer)
                action = self.agent.act_eps_greedy(s1)
                action_idx = action.index(1)
                self.game.make_action(action)
                reward,_ = self.get_reward()
                isterminal = self.game.is_episode_finished()
                s2 = self.preprocess(self.game.get_state().screen_buffer) if not isterminal else None
                
                transitions[step%10]=transition.Transition(s1,action_idx,s2,reward,isterminal,False)
                
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
            self.agent.batch_learn()
            if frames%FREQ_COPY:
                self.network.copy_params()
                
            if frames % FREQ_TEST == 0:
                steps_test,r,r_d,f,d = self.test_score()
                print("\t TEST at ", frames)
                print("\t SPEND STEPS:",steps_test)
                print("\tFRAG:",f,"DEATH:",d,"REWARD:",r)
                print("\t",r_d)
                self.agent.q_network.write_score(step=frames,reward=r,frag=f,death=d)
                self.start_episode()
                self.reward_gen.new_episode(health = self.game.get_game_variable(GameVariable.HEALTH),                                             ammo = self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO),                                             posx = self.game.get_game_variable(GameVariable.POSITION_X),                                             posy = self.game.get_game_variable(GameVariable.POSITION_Y))

        print(self.name," finished")
        self.finished = True




# In[4]:


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


# In[5]:


class Agent(object):

    def __init__(self,q_network,replay_buff,reward_gen):
        self.q_network = q_network
        self.replay_buff = replay_buff
        
        self.reward_repeat = 0

        self.record_score = 0

    def preprocess(self,img):
        if len(img.shape) == 3:
            img = img.transpose(1,2,0)

        img = skimage.transform.resize(img, RESOLUTION)
        img = img.astype(np.float32)
        return img

    def reset_reward_generator(self,reward_gen):
        self.reward_gen = reward_gen

    def margin(self, a_demo):
        ret = np.ones((BATCH_SIZE,N_ACTION),dtype=np.float32) * L_MIN
        ret[:,a_demo] = 0.0

        return ret
    
    def batch_learn(self):

        tree_idx, batch, is_weight = self.replay_buff.sample(BATCH_SIZE)

        s1 = np.zeros((N_ADV,BATCH_SIZE)+RESOLUTION,dtype=np.float32)
        s2 = np.zeros((N_ADV,BATCH_SIZE)+RESOLUTION,dtype=np.float32)
        actions = np.zeros((N_ADV,BATCH_SIZE,),dtype=np.int8)
        rewards = np.zeros((N_ADV,BATCH_SIZE,),dtype=np.float32)
        isterminals = np.zeros((N_ADV,BATCH_SIZE,),dtype=np.int8)
        isdemos = np.zeros((N_ADV,BATCH_SIZE,),dtype=np.int8)

        for i in range(BATCH_SIZE):
            for j in range(N_ADV):
                isterminals[j][i] = batch[i][j].isterminal
                isdemos[j][i] = batch[i][j].isdemo
                if isterminals[j][i] == False:
                    s1[j][i] = batch[i][j].s1
                    s2[j][i] = batch[i][j].s2
                    actions[j][i] = batch[i][j].action
                    rewards[j][i] = batch[i][j].reward
                else:
                    if type(batch[i][j].s1) != type(None):
                        s1[j][i] = batch[i][j].s1
                        actions[j][i] = batch[i][j].action
                        rewards[j][i] = batch[i][j].reward

        q2 = self.q_network.get_q_target_values(s2[0][:])
        target_q = self.q_network.get_q_values(s1[0][:])

        target_q_nsteps = np.empty((BATCH_SIZE, N_ACTION),dtype=np.float32)
        q_demo = np.empty((BATCH_SIZE, N_ACTION),dtype=np.float32)
        target_q_nsteps[:,:] = target_q
        q_demo[:,:] = target_q

        td_error = rewards[0] + DISCOUNT * (1-isterminals[0]) * q2

        target_q[range(target_q.shape[0]),actions[0]] = td_error

        target_q_nsteps[range(target_q_nsteps.shape[0]),actions[0]] =                         DISCOUNT*(1-isterminals[N_ADV-1]) * self.q_network.get_q_target_values(s2[N_ADV-1])
        for i in range(N_ADV-2,-1,-1):
            target_q_nsteps[range(target_q_nsteps.shape[0]),actions[0]] =                         (1-isterminals[i])*(rewards[i] + DISCOUNT*target_q_nsteps[range(target_q_nsteps.shape[0]),actions[0]]) +                         (isterminals[i])*(self.q_network.get_q_target_values(s2[i]))

        loss_class = (isdemos[0])*(np.amax((q_demo + self.margin(actions[0])), axis=1) - q_demo[np.arange(q_demo.shape[0]),actions[0]])

        self.q_network.learn(s1[0], target_q, target_q_nsteps, loss_class, is_weight)

    def perform_pre_learning_step(self):

        self.batch_learn()

        if frames % FREQ_COPY == 0:
            self.q_network.copy_params()

    def act_eps_greedy(self,s1):
        
        global frames
            
        if frames<TOTAL_STEPS*LINEAR_EPS_START:
            eps = EPS_START
        elif frames>=TOTAL_STEPS*LINEAR_EPS_START and frames<TOTAL_STEPS*LINEAR_EPS_END:
            eps = EPS_START + frames*(EPS_END-EPS_START)/(TOTAL_STEPS)
        else:
            eps = EPS_END
         
        ret_action = np.zeros((N_ACTION,))
        if random() <= eps:
            s1 = np.array(([s1]))
            a_idx = self.q_network.get_best_action(s1)
        else:
            a_idx = randint(0,N_ACTION-1)
        
        ret_action[a_idx] = 1
        return ret_action.tolist()
    
    def act_greedy(self,s1):
        ret_action = np.zeros((N_ACTION,))
        s1 = np.array(([s1]))
        a_idx = self.q_network.get_best_action(s1)        
        ret_action[a_idx] = 1
        return ret_action.tolist()

    def restore_model(self, model_path):
        self.q_network.restore_model(model_path)

    def save_model(self, model_path):
        self.q_network.save_model(model_path)


# In[6]:


class NetworkSetting:
    
    def conv1(pre_layer):
        num_outputs = 32
        kernel_size = [6,6]
        stride = [3,3]
        padding = 'SAME'
        activation = tf.nn.relu
        weights_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(0.1)
        
        return tf.contrib.layers.conv2d(pre_layer,kernel_size=kernel_size,                                        num_outputs=num_outputs,                                        stride=stride,padding=padding,activation_fn=activation,                                        weights_initializer=weights_init,                                        biases_initializer=bias_init)
    
    def maxpool1(pre_layer):
        return tf.nn.max_pool(pre_layer,[1,3,3,1],[1,2,2,1],'SAME')
    
    def conv2(pre_layer):
        num_outputs = 64
        kernel_size = [3,3]
        stride = [2,2]
        padding = 'SAME'
        activation = tf.nn.relu
        weights_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(0.1)
        return tf.contrib.layers.conv2d(pre_layer,kernel_size=kernel_size,num_outputs=num_outputs,                                        stride=stride,padding=padding,activation_fn=activation,                                        weights_initializer=weights_init,biases_initializer=bias_init)
    
    def maxpool2(pre_layer):
        return tf.nn.max_pool(pre_layer,[1,3,3,1],[1,2,2,1],'SAME')
        
    def reshape(pre_layer):
        return tf.contrib.layers.flatten(pre_layer)
        
    def fc1(pre_layer):
        num_outputs = 512
        activation_fn = tf.nn.relu
        weights_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.1)
        return tf.contrib.layers.fully_connected(pre_layer,num_outputs=num_outputs,activation_fn=activation_fn,                                                 weights_initializer=weights_init, biases_initializer=bias_init)
    
    def q_value(pre_layer):
        num_outputs = N_ACTION
        activation_fn = None
        weights_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.1)
        return tf.contrib.layers.fully_connected(pre_layer,num_outputs=num_outputs,activation_fn=activation_fn,                                                 weights_initializer=weights_init, biases_initializer=bias_init)


# In[7]:


class Network(object):

    def __init__(self):

        self.s1_ = tf.placeholder(tf.float32, [None] + [RESOLUTION[0],RESOLUTION[1], RESOLUTION[2]], name="State")
        self.a_ = tf.placeholder(tf.int32, [None], name="Action")
        self.target_q_one_ = tf.placeholder(tf.float32, [None, N_ACTION], name="TargetQ_one")
        self.target_q_n_ = tf.placeholder(tf.float32, [None, N_ACTION], name="TargetQ_n")
        self.loss_class_ = tf.placeholder(tf.float32,[None], name="Loss_Classification")

        self.is_weight_ = tf.placeholder(tf.float32,[None],name="IS_weight")
        
        with tf.device("/gpu:0"):
            with tf.variable_scope("learning"):
                self.model = self._model(True)
            with tf.variable_scope("target"):
                self.model_t = self._model(False)

            self._graph()
        
        with tf.device("/cpu:0"):
            self._summary()
    
    def _model(self,isLearning):
        if isLearning:
            self.conv1 = NetworkSetting.conv1(self.s1_)
            maxpool1 = NetworkSetting.maxpool1(self.conv1)
        else:
            conv1 = NetworkSetting.conv1(self.s1_)
            maxpool1 = NetworkSetting.maxpool1(conv1)
        if isLearning:
            self.conv2 = NetworkSetting.conv2(maxpool1)
            maxpool2 = NetworkSetting.maxpool2(self.conv2)
        else:
            conv2 = NetworkSetting.conv2(maxpool1)
            maxpool2 = NetworkSetting.maxpool2(conv2)
        reshape = NetworkSetting.reshape(maxpool2)
        fc1 = NetworkSetting.reshape(reshape)
        return NetworkSetting.q_value(fc1)
    
    def _graph(self):

        self.best_action = tf.argmax(self.model,1)
        
        self.weights_params_learning = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="learning")
        self.weights_params_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target")

        self.loss_one = tf.losses.mean_squared_error(self.model,self.target_q_one_)
        self.loss_n = tf.scalar_mul(tf.constant(LAMBDA1), tf.losses.mean_squared_error(self.model, self.target_q_n_))
        self.loss_class = tf.scalar_mul(tf.constant(LAMBDA2),tf.reduce_mean(self.loss_class_))
        self.loss_l2 = tf.scalar_mul(tf.constant(LAMBDA3),tf.add_n(self.get_l2_loss()))        

        self.loss = tf.add_n([self.loss_one,self.loss_n,self.loss_class,self.loss_l2])

        self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE)

        self.train_step = self.optimizer.minimize(tf.multiply(self.is_weight_,self.loss))
        
        self.copy_learn2target = [t_p.assign(l_p) for l_p,t_p in zip(self.weights_params_learning,self.weights_params_target)]
    
    def _summary(self):
        
        self.reward_ = tf.placeholder(tf.float32)
        self.frag_ = tf.placeholder(tf.int64)
        self.death_ = tf.placeholder(tf.int64)

        self.saver = tf.train.Saver()
        
        summary_lo = tf.summary.scalar('loss_one',self.loss_one)
        summary_ln = tf.summary.scalar('loss_n', self.loss_n)
        summary_ll = tf.summary.scalar('loss_l2', self.loss_l2)
        summary_lc = tf.summary.scalar('loss_class',self.loss_class)

        self.merged_scalar = tf.summary.merge([summary_lo,summary_ln,summary_ll,summary_lc])

        state_shape = self.s1_.get_shape()
        conv1_shape = self.conv1.get_shape()
        conv2_shape = self.conv2.get_shape()
        summary_state  = tf.summary.image('state',tf.reshape(self.s1_,[-1, state_shape[1], state_shape[2], state_shape[3]]),1)
        summary_conv1 = tf.summary.image('conv1',tf.reshape(self.conv1,[-1, conv1_shape[1], conv1_shape[2], 1]),1)
        summary_conv2 = tf.summary.image('conv2',tf.reshape(self.conv2,[-1, conv2_shape[1], conv2_shape[2], 1]),1)

        self.merged_image = tf.summary.merge([summary_state,summary_conv1,summary_conv2])
        
        summary_reward = tf.summary.scalar('reward',self.reward_)
        summary_frag = tf.summary.scalar('frag',self.frag_)
        summary_death = tf.summary.scalar('death',self.death_)
        
        self.merged_testscore = tf.summary.merge([summary_reward,summary_frag,summary_death])
        
        self.writer = tf.summary.FileWriter(LOG_DIR,SESS.graph)


    def learn(self, s1, target, target_n, loss_class,is_weight):

        global frames
        
        l, _ = SESS.run([self.loss, self.train_step],         feed_dict={self.s1_:s1, self.target_q_one_:target, self.target_q_n_:target_n, self.loss_class_:loss_class,         self.is_weight_:is_weight})
        
        self.write_loss_img(frames,s1,target,target_n,loss_class)

    def get_q_values(self, state):
        return SESS.run(self.model, feed_dict={self.s1_:state})

    def get_q_target_values(self,state):
        best_actions = self.get_best_actions(state)
        q =  SESS.run(self.model_t, feed_dict={self.s1_:state})
        ret = []
        for i in range(len(q)):
            ret.append(q[i][best_actions[i]])
        ret = np.array(ret)
        return ret

    def get_best_actions(self,state):
        s1 = state.reshape([-1,RESOLUTION[0],RESOLUTION[1],RESOLUTION[2]])
        return SESS.run(self.best_action, feed_dict={self.s1_:s1})

    def get_best_action(self,state):
        return SESS.run(self.best_action, feed_dict={self.s1_:state})[0]

    def write_loss_img(self,step,s1,target,target_n,loss_class):
        if step%100==0:
            feeddict = {self.s1_:s1,self.target_q_one_:target,self.target_q_n_:target_n,self.loss_class_:loss_class}
            m_l,m_i = SESS.run([self.merged_scalar,self.merged_image],feeddict)
            self.writer.add_summary(m_l,step)
            self.writer.add_summary(m_i,step)
            
    def write_score(self,step,reward,frag,death):
        feeddict={self.reward_:reward,self.frag_:frag,self.death_:death}
        m_r = SESS.run(self.merged_testscore,feeddict)
        self.writer.add_summary(m_r,step)

    def save_model(self, model_path):
        self.saver.save(SESS, model_path)

    def restore_model(self,model_path):
        self.saver.restore(SESS,model_path)

    def copy_params(self):

        self.copyop = [tf.assign(target, origin) for origin,target in zip(self.weights_params_learning,self.weights_params_target) ]
        SESS.run(self.copyop)

    def get_l2_loss(self):
        return [tf.nn.l2_loss(w) for w in self.weights_params_learning]


# In[8]:


frames = 0
SESS = tf.Session()

env = Environment("learning_1")
env.load_demonstration()

SESS.run(tf.global_variables_initializer())

print("---PRE TRAING PHASE---")
env.pre_learning()
    
print("---TRAINING PHASE---")
env.run()

