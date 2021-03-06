#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
from tqdm import tqdm
import network
import network_contrib
import network_double
import reward_generater
import replay_memory
import math

# Q-learning settings
learning_rate = 0.0025
discount_factor= 0.99
resolution = (30, 45, 3)
n_epoch = 2
steps_per_epoch = 100
testepisodes_per_epoch = 100
episode_timeout = 2000

frame_repeat = 12

capacity = 10**4

batch_size = 64

bots_num = 5

# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    #game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    name = "SampleRandomAgent"
    color = 0
    game.add_game_args("+name {} +colorset {}".format(name, color))
    #game.set_episode_timeout(episode_timeout)
    game.init()
    print("Doom initialized.")
    return game

def perform_learning_step(epoch,step,network,replay_memory,reward_gen):

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * n_epoch  # 10% of learning time
        eps_decay_epochs = 0.6 * n_epoch  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps
    
    #print(game.is_episode_finished(),":",end="")
    s1 = preprocess(game.get_state().screen_buffer)

    eps = exploration_rate(epoch)
    if random() <= eps:
        #print(len(actions))
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        a = network.get_best_action(np.array([s1]))
    reward = game.make_action(actions[a],frame_repeat)
    reward_gen.update_reward()
    reward = reward_gen.get_reward()

    isterminal = game.is_episode_finished()
    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

    # Remember the transition that was just experienced.
    replay_memory.add_transition(s1, a, s2, isterminal, reward)

    if replay_memory.size >batch_size:
        s1, a, s2, isterminal, r = replay_memory.get_sample(batch_size)

        q2 = network.get_q_target_values(s2)
        #q2 = np.max(network.get_q_target_values(s2), axis=1)
        #q2 = np.max(network.get_q_values(s2), axis=1)
        target_q = network.get_q_values(s1)

        target_q[np.arange(target_q.shape[0]),a] = r + discount_factor * (1-isterminal) * q2
        #print(target_q)
        network.learn(s1, target_q, reward, epoch*steps_per_epoch + step)
    

# Converts and down-samples the input image
def preprocess(img):
    if img.shape != resolution:
        img = img.transpose(1,2,0)

    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img

if __name__=="__main__":
    config_file_path = "./config/custom_config.cfg"
    #config_file_path = "./config/simpler_basic.cfg"
    ckpt_path = "./model/"
    game = initialize_vizdoom(config_file_path)

    print("learning rate: %f" % learning_rate)
    print("discount_factor %f" % discount_factor)
    print("resolution:",resolution)
    print("frame_repeat: %d" % frame_repeat)
    print("capacity:",capacity)
    print("barch_size: %d" % batch_size)
    print("screen_format:",game.get_screen_format())
    n_actions = game.get_available_buttons_size()
    actions = np.eye(n_actions,dtype=np.int32).tolist()
    print("action_size : %d" % (n_actions))
    #actions = [list(a) for a in it.product([0,1], repeat=n_actions)]

    replay_memory = replay_memory.ReplayMemory(resolution,capacity)

    session = tf.Session()
    network = network_double.network_simple(session,resolution,n_actions, learning_rate)
    #network = network.network_simple(session,resolution,n_actions, learning_rate)
    #network = network_contrib.network_contrib(session,resolution,n_actions,learning_rate)

    session.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):

        print("Epoch %d \n -----" % (epoch))
        print("Training Phase")
        train_episodes_finished = 0
        train_scores = []
        total_train_scores = []

        if epoch == 0:
            network.save_model("./models/model_00.ckpt")
        elif epoch == 5:
            network.save_model("./models/model_05.ckpt")
        else:
            pass

        game.new_episode()
        frag_count = game.get_game_variable(GameVariable.FRAGCOUNT)
        death_count = game.get_game_variable(GameVariable.DEATHCOUNT)
        print("FRAG: %d, DEATH: %d" % (frag_count, death_count))
        # Add bots
        for i in range(bots_num):
            game.send_game_command("addbot")
        reward_gen = reward_generater.reward_generater(game)

        for step in tqdm(range(steps_per_epoch)):

            if step%20 == 0:
                print(game.get_game_variable(GameVariable.POSITION_X))
                print(game.get_game_variable(GameVariable.POSITION_Y))

            if game.is_player_dead():
                game.respawn_player()
                reward_gen.reset_position()

            if game.is_episode_finished():
                score = reward_gen.get_total_reward()
                total_train_scores.append(score)
                game.new_episode()
                frag_count = game.get_game_variable(GameVariable.FRAGCOUNT)
                death_count = game.get_game_variable(GameVariable.DEATHCOUNT)
                print("FRAG: %d, DEATH: %d" % (frag_count, death_count))
                train_episodes_finished += 1
                print(game.get_episode_timeout())
                for i in range(bots_num):
                    game.send_game_command("addbot")
            
            perform_learning_step(epoch,step,network,replay_memory,reward_gen)

            if step % 10 == 0:
                score = reward_gen.get_reward()
                train_scores.append(score)

            if  step % 30 == 0:
                network.copy_params()

        #network.save_model(ckpt_path+"model.ckpt",epoch)
        frag_count = game.get_game_variable(GameVariable.FRAGCOUNT)
        death_count = game.get_game_variable(GameVariable.DEATHCOUNT)

        print("%d training episodes played." % train_episodes_finished)

        print("FRAG: %d, DEATH: %d" % (frag_count, death_count))

        train_scores = np.array(train_scores)
        if len(total_train_scores)==0:
            total_train_scores.append(0)
        total_train_scores = np.array(total_train_scores)
        print("Results: mean: %.1f(+-)%.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())
        print("Total Results: mean %.1f(plusminus)%.1f," %(total_train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % total_train_scores.min(), "max: %.1f," % total_train_scores.max())

    network.save_model("./models/model09.ckpt")
    print("Test Phase")

    test_scores=[]
    test_total_scores=[]

    game.new_episode()
    frag_count = game.get_game_variable(GameVariable.FRAGCOUNT)
    death_count = game.get_game_variable(GameVariable.DEATHCOUNT)
    print("FRAG: %d, DEATH: %d" % (frag_count, death_count))
    # Add bots
    for i in range(bots_num):
        game.send_game_command("addbot")
    reward_gen = reward_generater.reward_generater(game)

    while not game.is_episode_finished():

        if game.is_player_dead():
            game.respawn_player()
            reward_gen.reset_position()

        state = preprocess(game.get_state().screen_buffer)
        best_action_index = network.get_best_action(state)
        game.make_action(actions[best_action_index], frame_repeat)

        reward_gen.update_reward()
        reward = reward_gen.get_reward()
        test_scores.append(reward)
    
    test_scores = np.array(test_scores)
    print("Results: mean: %.1f(+-)%.1f," % (test_scores.mean(), test_scores.std()), \
                  "min: %.1f," % test_scores.min(), "max: %.1f," % test_scores.max())
    
    frag_count = game.get_game_variable(GameVariable.FRAGCOUNT)
    death_count = game.get_game_variable(GameVariable.DEATHCOUNT)
    print("FRAG: %d, DEATH: %d" % (frag_count, death_count))
    print("total: %d"%(reward_gen.get_total_reward()))

    game.close()