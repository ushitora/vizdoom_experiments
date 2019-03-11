#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import skimage.color, skimage.transform
from vizdoom import *
import os, time, random, threading, h5py, math,pickle, sys
import tensorflow as tf
import numpy as np
# from global_constants import *
from datetime import datetime, timedelta, timezone
from PIL import Image
# %matplotlib inline


# In[ ]:


class Environment(object):
    def __init__(self,sess,  name, game_instance, network, agent, replaymemory,start_time=None, end_time=None, n_step=None,  random_seed=0, position_data=None, parameters=None):
#     def __init__(self,sess,  name, start_time, end_time, parameter_server):
        self.name = name
        self.sess = sess
        self.parameters = parameters
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
        
        self.replay_memory = replaymemory
        
        self.step = 0
        self.model_gen_count = 0
        
        self.times_act = None
        self.times_update = None
        
        self.count_update = 0
        self.rewards_detail = None
        self.position_data_buff = position_data
        
        self.record_action = []
        self.record_treeidx = []
        
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
                play_log = []
                reward,frag, death,kill,total_detail,steps = self.test_agent(reward_buff =play_log)
                if self.parameters.save_data == True:
                    with open(os.path.join(self.parameters.play_lodir, "playlog_step%02d.txt"%int(self.progress*100)), 'wb') as f:
                        pickle.dump(play_log, f)
                if self.rewards_detail is not None:
                    self.rewards_detail.append(total_detail)

                if self.log_server is not None:
                    if kill <= 0:
                        steps = 100
                    self.log_server.write_score(self.sess,self.step,  reward, frag, death ,kill, steps, time.process_time())
                    self.log_server.write_processtime_score(self.sess, self.step, time.process_time())
                    if self.progress >= self.model_gen_count/12:
                        self.model_gen_count += 1
                        if self.parameters.save_data == True:
                            self.log_server.save_model(sess=self.sess, model_path=self.parameters.model_path, step=self.model_gen_count+1)
                
                self.step += 1
                if self.n_step is not None:
                    self.progress = self.step/self.n_step
                else:
                    self.progress = (datetime.now().timestamp() - self.start_time)/(self.end_time - self.start_time)
                if self.progress >= 1.0:
                    break
        except Exception as e:
            print(e)
            print(self.name, "killed ")
#             coordinator.request_stop(e)

    def learning_step(self):
        if self.step % self.parameters.interval_pull_params == 0:
            self.network.pull_parameter_server(self.sess)
        loss_values = []
        if not self.game.is_episode_finished() and self.game.get_screen_buff() is not None:
            
            if self.times_act is not None:
                start_time = datetime.now().timestamp()

            s1_ = self.preprocess(self.game.get_screen_buff())
            self.push_obs(s1_)
            agent_action_idx = self.agent.act_eps_greedy(self.sess, self.obs['s1'], self.progress)
            self.record_action.append(agent_action_idx)
            
            if self.position_data_buff is not None:
                enemy_label = self.game.get_label("Cacodemon")
                if enemy_label is not None:
                    center_x = enemy_label.x + enemy_label.width/2
                    center_y = enemy_label.y + enemy_label.height/2
#                     enemy_position_class = 2
                else:
                    center_x = 0
                    center_y = 0
#                     enemy_position_class = 0
                player_position_x = self.game.get_pos_x()
                player_position_y = self.game.get_pos_y()
                player_angle = self.game.get_angle()
                self.position_data_buff.append([center_x, center_y, player_position_x, player_position_y, player_angle])
            
#             engin_action = self.convert_action_agent2engine_simple(agent_action_idx)
            engin_action = self.convert_action_agent2engine(agent_action_idx)
            r,r_detail = self.game.make_action(self.step,engin_action , self.parameters.frame_repeat)
            isterminal = self.game.is_episode_finished()
            if isterminal:
                s2_ = np.zeros(self.parameters.resolution)
            else:
                s2_ = self.preprocess(self.game.get_screen_buff())
            
            self.push_batch( self.obs['s1'], agent_action_idx, s2_, r , isterminal, False)
            
            if self.times_act is not None:
                self.times_act.append(datetime.now().timestamp() - start_time)
            
            if len(self.memory) >= self.parameters.n_adv or isterminal:
                batch = self.make_advantage_data()
                self.clear_batch()
                for i,b in enumerate(batch):
                    if len(b) == 8:
                        self.replay_memory.store(b)
            
            self.step += 1
            
            if self.step % self.parameters.interval_update_network == 0:
                self.network.copy_network_learning2target(self.sess)
                
            if self.times_update is not None:
                start_time = datetime.now().timestamp()
            
            if self.step % self.parameters.interval_batch_learning == 0 and len(self.replay_memory) >= self.parameters.n_batch:
                s1, actions, r_one, r_adv, isdemo, is_weight, tree_idx = self.make_batch()
                self.record_treeidx.append(tree_idx)
                if self.log_server is not None:
                    self.count_idx[tree_idx] += 1
                loss_values = self.network.update_parameter_server(self.sess, s1, actions, r_one, r_adv, isdemo, is_weight)
                self.count_update += 1
                tderror = loss_values[4]
                l_one, l_n, l_m, l_l = loss_values[:-1]
                if self.log_server is not None:
                    self.log_server.write_loss(self.sess,self.step ,np.mean(l_one), np.mean(l_n), np.mean(l_m), l_l,time.process_time())
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

        s1, actions, r_one, r_adv, isdemo, is_weight, tree_idx = self.make_batch()
        loss_values = self.network.update_parameter_server(self.sess, s1, actions, r_one, r_adv, isdemo, is_weight)
        tderror = loss_values[4]
        l_one, l_n, l_m, l_l = loss_values[:-1]
        self.replay_memory.batch_update(tree_idx, tderror)
        
        if self.step % self.parameters.interval_update_network == 0:
            self.network.copy_network_learning2target(self.sess)
        
        if self.log_server is not None:
            if self.step % 10 == 0:
                self.log_server.write_loss(self.sess, self.step, np.mean(l_one), np.mean(l_n), np.mean(l_m), l_l,time.process_time())
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
            engine_action = self.convert_action_agent2engine(action)
            reward,reward_detail = self.game.make_action(step,engine_action,self.parameters.frame_repeat)
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
        for i in range(self.parameters.n_action):
            ans.append(agent_action%2)
            agent_action = int(agent_action / 2)
        return ans
    
    def convert_action_agent2engine_simple(self, agent_action):
        assert type(agent_action) == type(int()) or type(agent_action) == type(np.int64()), print("type(agent_action)=",type(agent_action))
        ans = np.zeros((self.parameters.n_agent_action,))
        ans[agent_action] = 1
        return ans.tolist()
    
    def preprocess(self,img):
        if len(img.shape) == 3 and img.shape[0]==3:
            img = img.transpose(1,2,0)

        img = skimage.transform.resize(img, self.parameters.resolution, mode="constant")
        img = img.astype(np.float32)
#         img = (img)/255.0
        return img

    def push_obs(self, s1):
        self.obs['s1'] = s1
        
    def clear_obs(self):
        self.obs = {}
        self.obs['s1'] = np.zeros(self.parameters.resolution, dtype=np.float32)
        
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
            R_adv = r + self.parameters.gamma*R_adv
            ret_batch.append(np.array([s1, a,s2,s2_adv,r ,R_adv ,isterminal, isdemo]))
        
        self.memory = []
        return ret_batch
    
    def make_batch(self):
        while True:
            tree_idx, batch_row, is_weight = self.replay_memory.sample(self.parameters.n_batch, self.calc_beta(self.progress))
#             tree_idx, batch_row, is_weight = self.replay_memory.sample(N_BATCH, 0.1)
            s2_input = [ batch_row[i,2] for i in range(self.parameters.n_batch)]
            s2_adv = [ batch_row[i,3] for i in range(self.parameters.n_batch)]
            if (np.shape(s2_input) == ((self.parameters.n_batch,)+self.parameters.resolution) and np.shape(s2_adv) == ((self.parameters.n_batch,)+self.parameters.resolution)):
                break
        
        s1, actions, s2, r_one, r_adv, isdemo = [],[],[],[],[],[]
        
        predicted_q_adv  = self.network.get_qvalue_max_learningaction(self.sess,s2_adv)
        
        predicted_q = self.network.get_qvalue_max_learningaction(self.sess,s2_input)
        
        for i in range(self.parameters.n_batch):
            s1.append(batch_row[i][0])
            actions.append(batch_row[i][1])
            R_one = batch_row[i][4] + self.parameters.gamma * predicted_q[i] if batch_row[i][6] == False else batch_row[i][4]
            R_adv = batch_row[i][5] + self.parameters.gamma**self.parameters.n_adv * predicted_q_adv[i] if batch_row[i][6] == False else batch_row[i][5]
            r_one.append(R_one)
            r_adv.append(R_adv)
            isdemo.append(batch_row[i][7])

        actions = np.array(actions)
        return s1, actions.astype(np.int32), r_one, r_adv, isdemo, is_weight, tree_idx
    
    def make_batch_uniform(self):
        while True:
            tree_idx, batch_row, is_weight = self.replay_memory.sample_uniform(N_BATCH)
            
            s2_input = [ batch_row[i,2] for i in range(self.parameters.n_batch)]
            s2_adv = [ batch_row[i,3] for i in range(self.parameters.n_batch)]
            if (np.shape(s2_input) == (self.parameters.n_batch,5, 120,120,3) and np.shape(s2_adv) == (self.parameters.n_batch,5, 120,120,3)):
                break
        
        s1, actions, s2, r_one, r_adv, isdemo = [],[],[],[],[],[]
        
        predicted_q_adv  = self.network.get_qvalue_max_learningaction(self.sess,s2_adv)
        
        predicted_q = self.network.get_qvalue_max_learningaction(self.sess,s2_input)
        
        for i in range(self.parameters.n_batch):
            s1.append(batch_row[i][0])
            actions.append(batch_row[i][1])
            R_one = batch_row[i][4] + self.parameters.gamma * predicted_q[i] if batch_row[i][6] == False else batch_row[i][4]
            R_adv = batch_row[i][5] + self.parameters.gamma**self.parameters.n_adv * predicted_q_adv[i] if batch_row[i][6] == False else batch_row[i][5]
            r_one.append(R_one)
            r_adv.append(R_adv)
            isdemo.append(batch_row[i][7])

        actions = np.array(actions)
        return s1, actions.astype(np.int32), r_one, r_adv, isdemo, is_weight, tree_idx
    
    def calc_beta(self, progress):
#         return BETA_MIN
        return (self.parameters.beta_max - self.parameters.beta_min) * progress + self.parameters.beta_min
    
    def exploring_step(self):
        if self.step % self.parameters.interval_pull_params == 0:
            self.network.pull_parameter_server(self.sess)
        loss_values = []
        if not self.game.is_episode_finished() and self.game.get_screen_buff() is not None:

            s1_ = self.preprocess(self.game.get_screen_buff())
            self.push_obs(s1_)
            agent_action_idx = self.agent.act_eps_greedy(self.sess, self.obs['s1'], self.progress)
#             engin_action = self.convert_action_agent2engine_simple(agent_action_idx)
            engin_action = self.convert_action_agent2engine(agent_action_idx)
            r,r_detail = self.game.make_action(self.step,engin_action , self.parameters.frame_repeat)
            isterminal = self.game.is_episode_finished()
            if isterminal:
                s2_ = np.zeros(self.parameters.resolution)
            else:
                s2_ = self.preprocess(self.game.get_screen_buff())
            
            self.push_batch( self.obs['s1'], agent_action_idx, s2_, r , isterminal, False)
            
            if len(self.memory) >= self.parameters.n_adv or isterminal:
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

