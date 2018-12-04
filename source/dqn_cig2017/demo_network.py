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
import reward_generater
import replay_memory
import math

# Q-learning settings
learning_rate = 0.00025
discount_factor= 0.99
resolution = (30, 45, 3)
n_epoch = 20
episodes_to_watch = 5

frame_repeat = 10

capacity = 2000

batch_size = 64

# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game

# Converts and down-samples the input image
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img

if __name__=="__main__":

    config_file_path = "./config/basic.cfg"
    ckpt_path = "./temp/"
    
    game = initialize_vizdoom(config_file_path)

    n_actions = game.get_available_buttons_size()
    actions = np.eye(n_actions,dtype=np.int32).tolist()

    print("%d actions is activate" % (n_actions))
    print(actions)

    replay_memory = replay_memory.ReplayMemory(resolution,capacity)

    session = tf.Session()
    network = network.network(session,resolution,n_actions, learning_rate)

    network.restore_model(ckpt_path+"model.ckpt")

    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = network.get_best_action(np.array([state]))
            best_action_index = best_action_index[0]

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)

