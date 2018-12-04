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
import math
#import network

# Q-learning settings
learning_rate = 0.00025
# learning_rate = 0.0001
discount_factor = 0.99
epochs = 1
learning_steps_per_epoch = 6000
replay_memory_size = 10**6

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 10

model_savefile = "/tmp/model.ckpt"
save_model = True
load_model = False
skip_learning = False
# Configuration file path
config_file_path = "../../scenarios/simpler_basic.cfg"

# Reward
rewards = {'living':-0.008, 'health_loss':-1, 'medkit':50, 'ammo':25, 'frag':100, 'dist':0.1}

# Converts and down-samples the input image
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img

class ReplayMemory:
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, resolution[0], resolution[1], channels)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, :, :, 0] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, 0] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]

def create_network(session, available_actions_count):
    # Create the input variables
    s1_ = tf.placeholder(tf.float32, [None] + list(resolution) + [1], name="State")
    a_ = tf.placeholder(tf.int32, [None], name="Action")
    target_q_ = tf.placeholder(tf.float32, [None, available_actions_count], name="TargetQ")

    # Add 2 convolutional layers with ReLu activation
    conv1 = tf.contrib.layers.convolution2d(s1_, num_outputs=8, kernel_size=[6, 6], stride=[3, 3],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
    conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=8, kernel_size=[3, 3], stride=[2, 2],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
    conv2_flat = tf.contrib.layers.flatten(conv2)
    fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=128, activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))

    q = tf.contrib.layers.fully_connected(fc1, num_outputs=available_actions_count, activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.1))
    best_a = tf.argmax(q, 1)

    loss = tf.losses.mean_squared_error(q, target_q_)

    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    # Update the parameters according to the computed gradient using RMSProp.
    train_step = optimizer.minimize(loss)

    with tf.name_scope('summary'):
        tf.summary.scalar('loss',loss)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs', session.graph)

    def function_learn(s1, target_q):
        feed_dict = {s1_: s1, target_q_: target_q}
        l, _, m = session.run([loss, train_step, merged], feed_dict=feed_dict)
        return l, m

    def function_get_q_values(state):
        return session.run(q, feed_dict={s1_: state})

    def function_get_best_action(state):
        return session.run(best_a, feed_dict={s1_: state})

    def function_simple_get_best_action(state):
        return function_get_best_action(state.reshape([1, resolution[0], resolution[1], 1]))[0]

    def function_write_loss(summary,step):
        writer.add_summary(summary,step)

    return function_learn, function_get_q_values, function_simple_get_best_action, function_write_loss

def learn_from_memory():
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)

        q2 = np.max(get_q_values(s2), axis=1)
        target_q = get_q_values(s1)
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2
        _,m = learn(s1, target_q)
        return m

    return None

def perform_learning_step(epoch):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    s1 = preprocess(game.get_state().screen_buffer)

    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        a = get_best_action(s1)
    reward = game.make_action(actions[a], frame_repeat)
    reward_gen.update_reward()
    reward = reward_gen.get_reward()

    isterminal = game.is_episode_finished()
    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, isterminal, reward)

    return learn_from_memory()
    #s1 = preprocess(game.get_state().screen_buffer)
class reward_generater(object):
    def __init__(self):
        self.pre_health = game.get_game_variable(GameVariable.HEALTH)
        self.pre_ammo = game.get_game_variable(GameVariable.AMMO0)
        self.pre_frag = 0
        self.pre_pos_x = 0
        self.pre_pos_y = 0
        self.reward = 0
    
    def update_reward(self):
        health = game.get_game_variable(GameVariable.HEALTH)
        ammo = game.get_game_variable(GameVariable.AMMO0)
        frag = game.get_game_variable(GameVariable.FRAGCOUNT)
        pos_x = game.get_game_variable(GameVariable.POSITION_X)
        pos_y = game.get_game_variable(GameVariable.POSITION_Y)

        if self.pre_pos_x == 0 and self.pre_pos_y == 0:
            self.pre_pos_x = pos_x
            self.pre_pos_y = pos_y
        
        self.reward = 0.0

        self.reward += (frag-self.pre_frag)*rewards['frag']
        self.reward += (math.sqrt((pos_x-self.pre_pos_x)**2 + (pos_y-self.pre_pos_x)**2)) * rewards['dist']
        if health > self.pre_health:
            self.reward += rewards['medkit']
        elif self.pre_health < health:
            self.reward += (self.pre_health-health)*rewards['health_loss']
        else:
            pass
        
        self.pre_frag = frag
        self.pre_health = health
        self.pre_ammo = ammo
        self.pre_pos_x = pos_x
        self.pre_pos_y = pos_y

    def reset_position(self):
        self.pre_pos_x = 0
        self.pre_pos_y = 0

    def get_reward(self):
        return self.reward

# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game


if __name__ == '__main__':

    bots_num = 10

    config_file_path = "./config/custom_config.cfg"
    game = initialize_vizdoom(config_file_path)

    n_actions = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0,1], repeat=n_actions)]

    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=replay_memory_size)

    session = tf.Session()
    learn, get_q_values, get_best_action, write_loss = create_network(session, len(actions))
    init = tf.global_variables_initializer()
    session.run(init)
    saver = tf.train.Saver()

    # Add bots
    for i in range(bots_num):
        game.send_game_command("addbot")

    time_start = time()

    for epoch in range(epochs):

        print("\nEpoch %d\n-------" % (epoch + 1))
        train_episodes_finished = 0
        train_scores = []

        print("Training...")
        game.new_episode()
        reward_gen = reward_generater()

        for learning_step in tqdm(range(learning_steps_per_epoch)):
            if game.is_player_dead():
                game.respawn_player()
                reward_gen.reset_position()

            m = perform_learning_step(epoch)

            if learning_step % 10 == 0:
              score = reward_gen.get_reward()
              train_scores.append(score)
              if m == None:
                  pass
              else:
                  write_loss(m, learning_step)

            if game.is_episode_finished():
                score = reward_gen.get_reward()
                train_scores.append(score)
                game.new_episode()
                train_episodes_finished += 1
        
        kill_count = game.get_game_variable(GameVariable.KILLCOUNT)
        frag_count = game.get_game_variable(GameVariable.FRAGCOUNT)
        death_count = game.get_game_variable(GameVariable.DEATHCOUNT)

        print("%d training episodes played." % train_episodes_finished)

        print("KILL: %d, FRAG: %d, DEATH: %d" % (kill_count, frag_count, death_count))

        train_scores = np.array(train_scores)

        print("Results: mean: %.1f(+-)%.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

    game.close()

"""
if __name__="__main__":

    bots_num = 10

    resolution = [30,30]

    config_file_path = "./config/custom_config.cfg"
    game = initialize_vizdoom(config_file_path)

    n_actions = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0,1], repeat=n_actions)]

    session = tf.Session()
    network = network(resolution,n_actions,session)

    game.new_episode()

    for step in tqdm(range(learning_steps_per_epoch)):

        s1 = preprocess(game.get_state().screen_buffer)

        network.test(s1)

    game.close()
"""