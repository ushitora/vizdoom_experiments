#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import sys
from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
from tqdm import tqdm
import replay_memory
import math
import skimage.color, skimage.transform
import transition

n_transit = 100
n_steps = 10

# Q-learning settings
learning_rate = 0.0025
discount_factor= 0.99
resolution = (30, 45, 1)
config_file_path = "./config/simpler_basic.cfg"
freq_copy = 30
frame_repeat = 5
capacity = 10**4
batch_size = 64

sleep_time = 0.028

def preprocess(img):
    if len(img.shape) == 3:
        img = img.transpose(1,2,0)

    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img

# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.set_mode(Mode.SPECTATOR)
    game.set_screen_format(ScreenFormat.GRAY8)
    #game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game

if __name__ == "__main__":
    args = sys.argv
    game = initialize_vizdoom(config_file_path)

    n_actions = game.get_available_buttons_size()
    actions = np.eye(n_actions,dtype=np.int32).tolist()
    replaymemory = replay_memory.ReplayMemory(n_transit)

    game.new_episode()
    total_rewards = []
    total_reward = 0

    transitionset = np.empty((n_steps,),dtype=object)

    state_num = 0

    # step = 0
    for t in tqdm(range(n_transit)):
        if game.is_episode_finished():
            total_rewards.append(total_reward)
            print("total_reward = ",total_reward)
            game.new_episode()
            total_reward = 0

        step = 0
        while step < n_steps:
            if not game.is_episode_finished():
                s1 = preprocess(game.get_state().screen_buffer)
                game.advance_action(frame_repeat)
                reward = game.get_last_reward()
                action = game.get_last_action()
                if action in actions:
                    total_reward += reward
                    print(reward,end=',')
                    idx = actions.index(action)
                    isterminal = game.is_episode_finished()
                    if not isterminal:
                        s2 = preprocess(game.get_state().screen_buffer)
                    else:
                        s2 = None
                
                    transitionset[step] = transition.Transition(s1,idx,s2,reward,isterminal,True)
                    step += 1
            else:
                transitionset[step] = transition.Transition(None,None,None,None,True,True)
                step += 1
            
        print("\n")
    
        replaymemory.store(np.copy(transitionset))

    total_reward = np.array(total_reward)
    print("Results: mean: %.1f(+-)%.1f," % (total_reward.mean(), total_reward.std()), \
                  "min: %.1f," % total_reward.min(), "max: %.1f," % total_reward.max())

    replaymemory.save_data()

    game.close()