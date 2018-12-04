#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from vizdoom import *
import tensorflow as tf
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
from tqdm import tqdm
import reward_generater
import replay_memory
import network_double
import agent_dqn_cig2017
import math

#Q-learning settings
n_epoch_pre = 10
steps_per_epoch_pre = 500

n_epoch = 10
steps_per_epoch = 1000

n_test_episodes = 5

n_q_steps = 10

learning_rate = 0.0025
discount_factor= 0.99

resolution = (120, 180, 3)

config_file_path = "./config/custom_config.cfg"
ckpt_path = "./model_v05/"
demo_path = "./demonstration/"
hdf_name = "demodata_cig2017.h5"

freq_copy = 30
frame_repeat = 5

capacity = 10**5

batch_size = 64

lambda1 = 1.0
lambda2 = 1.0
lambda3 = 10.0e-5

n_bot = 5

freq_test = 100
#%%
# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    #game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game
    
#%%
if __name__=="__main__":
    game = initialize_vizdoom(config_file_path)

    print("Epoch:%d, steps:%d"%(n_epoch,steps_per_epoch))
    print("learning rate: %f" % learning_rate)
    print("discount_factor %f" % discount_factor)
    print("resolution:",resolution)
    print("frame_repeat: %d" % frame_repeat)
    print("capacity:",capacity)
    print("barch_size: %d" % batch_size)
    print("screen_format:",game.get_screen_format())
    print("lambda1:%f, lambda2:%f, lambda3:%f" % (lambda1,lambda2,lambda3))
    n_actions = game.get_available_buttons_size()
    actions = np.eye(n_actions,dtype=np.int32).tolist()

    replay_memory = replay_memory.ReplayMemory(capacity,data_name="demodata_cig2017.npy")

    config=tf.ConfigProto(log_device_placement=False)
    config.allow_soft_placement=True
    session = tf.Session(config=config)
    network = network_double.network_simple(session,resolution,n_actions,learning_rate)

    session.run(tf.global_variables_initializer())

    reward_gen = reward_generater.reward_generater(game)

    agent = agent_dqn_cig2017.agent_dqn(n_epoch,
                                network,
                                replay_memory,
                                reward_gen,
                                actions,
                                resolution,
                                n_q_steps,
                                discount_factor,
                                learning_rate,
                                frame_repeat,
                                batch_size,
                                freq_copy,
                                lambda1,lambda2,lambda3)

    
    agent.restore_model(ckpt_path+"pre_model.ckpt")
    
    for i in range(5):
        print("--------------------------\nTest",i)
        pre_frag_count = game.get_game_variable(GameVariable.FRAGCOUNT)
        pre_death_count = game.get_game_variable(GameVariable.DEATHCOUNT)

        game.new_episode()
        for i in range(n_bot):
            game.send_game_command("addbot")
        agent.reward_gen.new_episode()
        
        step = 0
        total_reward = 0.0
        while not game.is_episode_finished():
            if game.is_player_dead():
                game.respawn_player()
                agent.reward_gen.respawn_pos(game.get_game_variable(GameVariable.POSITION_X),game.get_game_variable(GameVariable.POSITION_Y))

            if game.is_episode_finished():
                break

            best_action_index = agent.get_best_action(game.get_state().screen_buffer)

            game.make_action(actions[best_action_index], frame_repeat)

            reward,reward_detail = agent.reward_gen.get_reward()
            total_reward += reward

            if step%10 == 0:
                print("\t-----------\n\tstep",step)
                print("\tFRAGCOUNT: %d, DEATHCOUNT: %d, POSX: %.2f POSY: %.2f REWARD: %.2f" \
                         % (game.get_game_variable(GameVariable.FRAGCOUNT),game.get_game_variable(GameVariable.DEATHCOUNT), \
                         game.get_game_variable(GameVariable.POSITION_X),game.get_game_variable(GameVariable.POSITION_Y), \
                         reward))
                print("\t",end="")
                for k,v in reward_detail.items():
                    print("%s: %.2f ,"%(k,v), end="")
                print("\n")

            step += 1

        frag_count = game.get_game_variable(GameVariable.FRAGCOUNT)
        death_count = game.get_game_variable(GameVariable.DEATHCOUNT)
        
        print("FRAG:%d, DEATH: %d, REWARD: %.3f" %(frag_count - pre_frag_count, death_count - pre_death_count, total_reward))
    
    game.close()
