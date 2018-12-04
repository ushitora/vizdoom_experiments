#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from multiprocessing import Process, Queue, Pipe, Value, Array, Manager, Pool , TimeoutError
import skimage.color, skimage.transform
from vizdoom import *
import os, time, random, threading
import tensorflow as tf
import numpy as np
from game_instance import GameInstance
from global_constants import *
from datetime import datetime, timedelta
from PIL import Image


# In[ ]:


LOGDIR = "./logs/multi_agent/log_test/"
MODEL_PATH =  "./models/multi_agent/model_test/model.ckpt"
__name__ = "learning"


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
        self.sess.run(tf.global_variables_initializer())
        self.log_server = None
        
        self.step = 0
        print(self.name," initialized...")
        
    def run_learning(self, coordinator):
        
        self.game.new_episode(BOTS_NUM)
        try:
            while not coordinator.should_stop():
                self.learning_step()
                self.progress = (datetime.now().timestamp() - self.start_time)/(self.end_time - self.start_time)
                if self.progress >= 1.0:
                    coordinator.request_stop()
                    break
        except Exception as e:
            coordinator.request_stop(e)
        return 0
    
    def run_test(self, coordinator):
        
        try:
            while not coordinator.should_stop():
                reward,frag, death = self.test_agent()
                print("----------TEST at",(datetime.now()), "---------")
                print("FRAG:",frag, "DEATH:",death)
                print("REWARD",reward)

                if self.log_server is not None:
                    self.log_server.write_score(self.sess,self.step,  reward, frag, death)
                self.step += 1
                time.sleep(60 * 2)
                self.progress = (datetime.now().timestamp() - self.start_time)/(self.end_time - self.start_time)
                if self.progress >= 1.0:
                    coordinator.request_stop()
        except Exception as e:
            coordinator.request_stop(e)
    
    def learning_step(self):
        l_p = 0
        l_v = 0
        if not self.game.is_episode_finished() and self.game.get_screen_buff() is not None:

            s1_ = self.preprocess(self.game.get_screen_buff())
            agent_action_idx = self.agent.act_eps_greedy(self.sess, s1_, self.progress)
            engin_action_idx = self.convert_action_agent2engine(agent_action_idx)
            engin_action = np.zeros((N_ENGINE_ACTION,), dtype=np.int32)
            engin_action[engin_action_idx] = 1
            r,_ = self.game.make_action(engin_action.tolist() , FRAME_REPEAT)
            isterminal = self.game.is_episode_finished()

            self.agent.push_batch(s1_, agent_action_idx, r, isterminal)

            if (self.game.is_player_dead()):
                    self.game.respawn_player()

            if self.agent.batch_pointer >= N_ADV-1:
                l_p, l_v = self.agent.train_network(self.sess)
                if self.log_server is not None:
                    if self.step % 10 == 0:
                        self.log_server.write_loss(self.sess,self.step, l_p[0][0], l_v[0][0])
                        self.log_server.write_weights(self.sess, self.step)
                    
                self.agent.clear_batch()
            
            self.step += 1
            
            self.network.pull_parameter_server(self.sess)
                
        else:
            l_p, l_v = self.agent.train_network(self.sess)
            self.game.new_episode(BOTS_NUM)
            self.agent.clear_batch()
            self.agent.clear_obs()
            
        return l_p, l_v
            
    def test_agent(self, gif_buff=None, reward_buff=None):
        
        self.game.new_episode(BOTS_NUM)
        
#         Copy params from global
        self.network.pull_parameter_server(self.sess)

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
            print(np.shape(gif_img))
            for i in range(len(gif_img)):
                save_img.append(Image.fromarray(np.uint8(gif_img[i])))
            gif_buff += save_img
        return total_reward, self.game.get_frag_count(), self.game.get_death_count()
        
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
            self.state1_ = tf.placeholder(tf.float32, shape=(None,) + RESOLUTION)
            self._build_model(self.state1_)

            self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="parameter_server")
            self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, RMSProbDecaly)
            
        with tf.variable_scope("summary", reuse=tf.AUTO_REUSE):
            self._build_summary(sess,log_dir)
        
        self.saver = tf.train.Saver()
        
#         print("-------GLOBAL-------")
#         for w in self.weights_params:
#             print(w)

    def _build_model(self,state):
            self.conv1 = NetworkSetting.conv1(state)
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
        
        with tf.variable_scope("Summary_Score"):
            s = [tf.summary.scalar('reward', self.reward_, family="score"), tf.summary.scalar('frag', self.frag_, family="score"), tf.summary.scalar("death", self.death_, family="score")]
            self.summary_reward = tf.summary.merge(s)
        
        with tf.variable_scope("Summary_Loss"):
            self.summary_loss = tf.summary.merge([tf.summary.scalar('loss_policy', self.loss_p_, family="loss"), tf.summary.scalar('loss_value', self.loss_v_, family="loss")])
        
        with tf.variable_scope("Summary_Images"):
            conv1_display = tf.reshape(tf.transpose(self.conv1, [0,3,1,2]), (-1, self.conv1.get_shape()[1],self.conv1.get_shape()[2]))
            conv2_display = tf.reshape(tf.transpose(self.conv2, [0,3,1,2]), (-1, self.conv2.get_shape()[1],self.conv2.get_shape()[2]))
            conv1_display = tf.expand_dims(conv1_display, -1)
            conv2_display = tf.expand_dims(conv2_display, -1)

            state_shape = self.state1_.get_shape()
            conv1_shape = conv1_display.get_shape()
            conv2_shape = conv2_display.get_shape()

            s_img = []
            s_img.append(tf.summary.image('state',tf.reshape(self.state1_,[-1, state_shape[1], state_shape[2], state_shape[3]]), 1, family="state1"))
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
    
    def write_loss(self,sess, step, l_p, l_v):
        m = sess.run(self.summary_loss, feed_dict={self.loss_p_: l_p, self.loss_v_:l_v})
        return self.writer.add_summary(m, step)
    
    def write_img(self,sess, step, state):
        m = sess.run(self.summary_image, feed_dict={self.state1_: state})
        return self.writer.add_summary(m, step)
    
    def write_weights(self, sess, step):
        m = sess.run(self.summary_weights)
        return self.writer.add_summary(m, step)
        
    def load_model(self, sess, model_path):
        self.saver.restore(sess, model_path)
    
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

        self.batch = {}
        self.batch['s1'] = np.zeros((N_ADV,) + RESOLUTION, dtype=np.float32)
        self.batch['action'] = np.zeros((N_ADV, ), dtype=np.float32)
        self.batch['reward'] = np.zeros((N_ADV,), dtype=np.float32)
        self.batch['isterminal'] = np.zeros((N_ADV, ), dtype=np.float32)
        self.batch_pointer = 0
        
    def calc_eps(self, progress):
        if progress < 0.2:
            return EPS_MIN
        elif progress >= 0.2 and progress < 0.8:
            return ((EPS_MAX - EPS_MIN)/ 0.6) * progress + EPS_MIN
        else :
            return EPS_MAX

    def act_eps_greedy(self, sess, s1, progress):
        assert progress >= 0.0 and progress <=1.0
        
        self.push_obs(s1)
        eps = self.calc_eps(progress)
        if random.random() > 0.5:
            p = self.network.get_policy(sess, self.obs['s1'])[0]
            a_idx = np.random.choice(N_AGENT_ACTION, p=p)
        else:
            a_idx = np.random.randint(N_AGENT_ACTION)
            
        return a_idx
    
    def act_greedy(self, sess, s1):
        p = self.network.get_policy(sess, self.obs['s1'])[0]
        a_idx = np.random.choice(N_AGENT_ACTION, p=p)
        return a_idx
    
    def get_gradients(self, sess):
        return self.network.get_gradients(sess, self.batch['s1'], self.batch['action'], self.batch['reward'], self.batch['isterminal'])
    
    def train_network(self, sess):
        r_adv = []
        for i in range(N_ADV):
            if self.batch['isterminal'][i] != True:
                if i == 0:
                    r_adv.append(self.batch['reward'][i])
                else:
                    r_adv.append(r_adv[-1]*GAMMA + self.batch['reward'][i])
            else:
                r_adv.append(0)
            
        return self.network.update_parameter_server(sess, self.batch['s1'], self.batch['action'], r_adv, self.batch['isterminal'])
    
    def push_obs(self, s1):
        self.obs['s1'] = np.roll(self.obs['s1'],shift=1, axis=0)
        self.obs['s1'][-1] = s1
        
    def clear_obs(self):
        self.obs = {}
        self.obs['s1'] = np.zeros((N_ADV,)+ RESOLUTION, dtype=np.float32)
        
    def push_batch(self, s1, action, reward, isterminal):
        self.batch['s1'][self.batch_pointer] = s1
        self.batch['action'][self.batch_pointer] =action
        self.batch['reward'][self.batch_pointer] =reward
        self.batch['isterminal'][self.batch_pointer] = isterminal
        self.batch_pointer += 1
    
    def clear_batch(self):
        self.batch['s1'] = np.zeros((N_ADV,) + RESOLUTION, dtype=np.float32)
        self.batch['action'] = np.zeros((N_ADV, ), dtype=np.float32)
        self.batch['reward'] = np.zeros((N_ADV,), dtype=np.float32)
        self.batch['isterminal'] = np.zeros((N_ADV, ), dtype=np.float32)
        self.batch_pointer = 0


# In[ ]:


class NetworkLocal(object):
    def __init__(self,name, parameter_server):
        self.name = name
        
        with tf.variable_scope(self.name+"_train", reuse=tf.AUTO_REUSE):
            self.state1_ = tf.placeholder(tf.float32,shape=(None,)+RESOLUTION, name="state_1")
            self.a_ = tf.placeholder(tf.float32, shape=(None,), name="action")
            self.r_adv = tf.placeholder(tf.float32, shape=(None,), name="reward_advantage")
            self.isterminal_ = tf.placeholder(tf.float32, shape=(None,), name="isterminal")
            self.policy, self.value, self.conv1, self.conv2 = self._model(self.state1_)

            self._build_graph(parameter_server)
            
            self.global_weights_ = [tf.placeholder(tf.float32, w.get_shape()) for w in self.weights_params]
            self.assign_weights = [l_p.assign(g_p) for l_p,g_p in zip(self.weights_params,self.global_weights_)]

    def _model(self,state):

#         with tf.variable_scope(self.name + "_nottrainable"):
        conv1 = NetworkSetting.conv1(state)
        maxpool1 = NetworkSetting.maxpool1(conv1)
        conv2 = NetworkSetting.conv2(maxpool1)
        maxpool2 = NetworkSetting.maxpool2(conv2)
        reshape = NetworkSetting.reshape(maxpool2)
        
        with tf.variable_scope(self.name + "_trainable"):
#             rnn ,l ,_ = NetworkSetting.lstm(reshape, state)
            fc1 = NetworkSetting.fc1(reshape)

            policy = NetworkSetting.policy(fc1)
            value = NetworkSetting.value(fc1)
        
        return policy, value, conv1, conv2

    def _build_graph(self, parameter_server):

        one_hot = tf.one_hot(tf.cast(self.a_, tf.int32), depth=N_AGENT_ACTION)
        
        log_prob = tf.log(tf.reduce_sum(self.policy * one_hot, axis=1, keep_dims=True)+1e-10)
        advantage = tf.reshape(self.r_adv, [-1,1]) - self.value
        self.loss_policy = -log_prob * tf.stop_gradient(advantage)
        self.loss_value = tf.square(advantage)
        self.loss_total = tf.reduce_mean(self.loss_policy + self.loss_value)
        
        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+"_train/")
        
        self.grads = tf.gradients(self.loss_total ,self.weights_params)
        
        self.update_global_weight_params =             parameter_server.optimizer.apply_gradients(zip(self.grads, parameter_server.weights_params))
        
        self.pull_global_weight_params = [l_p.assign(g_p) for l_p,g_p in zip(self.weights_params,parameter_server.weights_params)]

        self.push_local_weight_params = [g_p.assign(l_p) for g_p,l_p in zip(parameter_server.weights_params,self.weights_params)]
        
    def pull_parameter_server(self, sess):
        sess.run(self.pull_global_weight_params)
    
    def push_parameter_server(self):
        sess.run(self.push_local_weight_params)
        
    def get_weights(self, sess):
        return sess.run(self.weights_params)
    
    def get_gradients(self,sess, s1, a, r, isterminal):
        assert np.ndim(s1) == 4
        
        feed_dict = {self.state1_: s1, self.a_:a, self.r_adv:r,  self.isterminal_:isterminal}
        return sess.run(self.grads, feed_dict)
    
    def update_parameter_server(self,sess, s1, a, r, isterminal):
        assert np.ndim(s1) == 4
        feed_dict = {self.state1_: s1,self.a_:a, self.r_adv:r}
        _, l_p, l_v = sess.run([self.update_global_weight_params, self.loss_policy, self.loss_value],feed_dict)
        return l_p, l_v
    
    def check_weights(self, sess):
        weights = SESS.run(self.weights_params)
        assert np.isnan([np.mean(w) for w in weights]).any()==False , print(weights)

    def get_policy(self, sess, s1):
        assert np.ndim(s1) == 4    
        return sess.run(self.policy, {self.state1_: s1})

    def get_value(self, sess, s1):
        assert np.ndim(s1) == 4        
        return sess.run(self.value, {self.state1_:s1})


# In[ ]:


class NetworkSetting:
    
    def conv1(pre_layer):
        num_outputs = 32
#         kernel_size = [1,6,6]
#         stride = [1,3,3]
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
        num_outputs = 32
#         kernel_size = [1,3,3]
#         stride = [1,2,2]
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
        shape = pre_layer.get_shape()
        return tf.reshape(pre_layer, shape=(-1, shape[1]*shape[2]*shape[3]))
    
    def fc1(pre_layer):
        num_outputs =1024
        activation_fn = tf.nn.relu
        weights_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.1)
        return tf.contrib.layers.fully_connected(pre_layer,num_outputs=num_outputs,activation_fn=activation_fn,weights_initializer=weights_init, biases_initializer=bias_init)
    
    def policy(pre_layer):
        num_outputs=N_AGENT_ACTION
        activation_fn = tf.nn.softmax
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


if __name__=="learning":
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list=USED_GPU))
    config.log_device_placement = False
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    with tf.device('/gpu:0'):
        parameter_server = ParameterServer(sess,LOGDIR)

        starttime = datetime.now().timestamp()
        end_time = (datetime.now() + timedelta(minutes=15)).timestamp()
        
        coordinator = tf.train.Coordinator()

        environments = []
        threads = []
        for i in range(2):
            name = "worker_%d"%(i)
            game_instance=GameInstance(DoomGame(), name=name, config_file_path=CONFIG_FILE_PATH, rewards=REWARDS,n_adv=N_ADV)
            network = NetworkLocal(name, parameter_server)
            agent = Agent(network)
            env = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, start_time=starttime, end_time=end_time)
            environments.append(env)
            
        environments[0].log_server = parameter_server

        name = "test"
        game_instance=GameInstance(DoomGame(), name=name, config_file_path=CONFIG_FILE_PATH, rewards=REWARDS,n_adv=N_ADV)
        network = NetworkLocal(name, parameter_server)
        agent = Agent(network)
        test_env = Environment(sess = sess ,name=name, agent=agent, game_instance=game_instance, network=network, start_time=starttime, end_time=end_time)
        test_env.log_server = parameter_server
        thread_test = threading.Thread(target=test_env.run_test, args=(coordinator,))
        
        parameter_server.write_graph(sess)

        print("-----Start ASYNC LEARNING----")
        for worker in environments:
            t = threading.Thread(target=worker.run_learning, args=(coordinator,) )
            threads.append(t)
        
        threads.append(thread_test)
        
        for t in threads:
            t.start()
        coordinator.join(threads)  
       


# In[ ]:


GIF_BUFF = []
test_env.test_agent(gif_buff=GIF_BUFF)
GIF_BUFF[0].save('gifs/test.gif',save_all=True, append_images=GIF_BUFF[1:], optimize=False, duration=40, loop=0)


# In[ ]:


environments[0].step


# In[ ]:




