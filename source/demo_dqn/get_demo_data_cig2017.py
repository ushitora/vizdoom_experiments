#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
from random import sample, randint, random,choice
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
from tqdm import tqdm
import network_double
import reward_generater
import replay_memory
import math
import transition
import h5py

MINUS = '\033[94m'
ENDC = '\33[0m'

# Q-learning settings
learning_rate = 0.0025
discount_factor= 0.99
resolution = (120, 180, 3)
n_epoch = 10
steps_per_epoch = 2000
testepisodes_per_epoch = 100
episode_timeout = 2000

frame_repeat = 4

capacity = 10**4

batch_size = 64

bots_num = 16

n_transit = 100

n_steps = 10

use_hdf5 = True

demo_path = "./demonstration/"
outputfile = "demodata_cig2017_v0-2.h5"

state1 = np.empty((n_transit,n_steps,resolution[0],resolution[1],resolution[2]),dtype=np.float32)
state2 = np.empty((n_transit,n_steps,resolution[0],resolution[1],resolution[2]),dtype=np.float32)
actions = np.empty((n_transit,n_steps,1),dtype=np.int8)
isterminals = np.empty((n_transit,n_steps,1),dtype=np.int8)

healths = np.empty((n_transit,n_steps,1),dtype=np.float32)
ammos = np.empty((n_transit,n_steps,1),dtype=np.float32)
frags = np.empty((n_transit,n_steps,1),dtype=np.float32)
deaths = np.empty((n_transit,n_steps,1),dtype=np.float32)
posxs = np.empty((n_transit,n_steps,1),dtype=np.float32)
posys = np.empty((n_transit,n_steps,1),dtype=np.float32)

def preprocess(img):
    if len(img.shape) == 3:
        img = img.transpose(1,2,0)

    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img

def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.set_mode(Mode.SPECTATOR)
    # game.set_mode(Mode.PLAYER)
    #game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    name = "YOU"
    color = 0
    game.add_game_args("+name {} +colorset {}".format(name, color))
    #game.set_episode_timeout(episode_timeout)
    game.init()
    print("Doom initialized.")
    return game

def print_variables(game):
    frag = game.get_game_variable(GameVariable.FRAGCOUNT)
    death = game.get_game_variable(GameVariable.DEATHCOUNT)
    ammo = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
    health = game.get_game_variable(GameVariable.HEALTH)
    posx = game.get_game_variable(GameVariable.POSITION_X)
    posy = game.get_game_variable(GameVariable.POSITION_Y)

    # print("ACTION:",game.get_last_action())
    print("FRAG:",frag," DEATH:",death," AMMO:",ammo, " HEALTH",health)
    print("POSX:%.4f POSY:%.4f" % (posx, posy))

def store_data(trans,step,s1,s2,action_idx,isterminal,health,ammo,frag,death,posx,posy):
    state1[trans][step] = s1
    state2[trans][step] = s2
    actions[trans][step] = action_idx
    isterminals[trans][step] = isterminal
    healths[trans][step] = health
    ammos[trans][step] = ammo
    frags[trans][step] = frag
    deaths[trans][step] = death
    posxs[trans][step] = posx
    posys[trans][step] = posy

if __name__=="__main__":

    game = initialize_vizdoom("./config/custom_config.cfg")

    n_actions = game.get_available_buttons_size()
    # actions = [list(a) for a in it.product([0, 1], repeat=n_actions)]
    commands = np.eye(n_actions,dtype=np.int32).tolist()
    replaymemory = replay_memory.ReplayMemory(n_transit,data_name="demodata_cig2017.npy")
    
    r_gen = reward_generater.reward_generater(game)

    # demo_data = replay_memory.ReplayMemory(resolution,n_transit)

    game.new_episode()

    for i in range(bots_num):
        game.send_game_command("addbot")

    total_reward = 0.0
    death_bias = 0
    for transit in tqdm(range(n_transit)):
        if game.is_episode_finished():
            print("***********************\n\tGAME IS FINISHED \n\tTOTAL:%d\n*****************************"%total_reward)
            total_reward = 0
            death_bias = game.get_game_variable(GameVariable.DEATHCOUNT)
            game.new_episode()
            r_gen.new_episode()
            for i in range(bots_num):
                game.send_game_command("addbot")

        if transit%2 == 0:
            r_gen.update_origin()
        step = 0
        while step < n_steps:
            if not game.is_episode_finished() and game.is_player_dead():
                game.respawn_player()
                r_gen.respawn_pos(game.get_game_variable(GameVariable.POSITION_X),game.get_game_variable(GameVariable.POSITION_Y))

            if not game.is_episode_finished():

                pre_frag = game.get_game_variable(GameVariable.FRAGCOUNT)
                pre_death = game.get_game_variable(GameVariable.DEATHCOUNT)
                pre_ammo = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
                pre_health = game.get_game_variable(GameVariable.HEALTH)
                pre_posx = game.get_game_variable(GameVariable.POSITION_X)
                pre_posy = game.get_game_variable(GameVariable.POSITION_Y)

                s1 = preprocess(game.get_state().screen_buffer)
                game.advance_action(frame_repeat)
                isterminal = game.is_episode_finished()
                action = game.get_last_action()
                if not action.count(1.0)==0:
                    r,_ = r_gen.get_reward()
                    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

                    print("-----------------------------")
                    print("State #",transit,"-",step)
                    new_action = [0.0 for i in range(n_actions)]
                    if not action.count(1.0)==1:
                        print("ACTION:",action,"->",end="")
                        for i in range(n_actions-1,-1,-1):
                            if action[i]==1.0:
                                new_action[i] = 1.0
                                break
                        print(new_action)
                        idx = commands.index(new_action)
                    else:
                        print("ACTION:",action)
                        idx = commands.index(action)

                    if r < 0:
                        print("REWARD:"+MINUS+str(r)+ENDC)
                    else:
                        print("REWARD:",r)
                    total_reward += r

                    frag = game.get_game_variable(GameVariable.FRAGCOUNT)
                    death = game.get_game_variable(GameVariable.DEATHCOUNT)
                    ammo = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
                    health = game.get_game_variable(GameVariable.HEALTH)
                    posx = game.get_game_variable(GameVariable.POSITION_X)
                    posy = game.get_game_variable(GameVariable.POSITION_Y)

                    print("FRAG: %d (%d)"%(frag,frag-pre_frag)," DEATH: %d (%d)"%(death-death_bias,death-pre_death)," AMMO: %d (%d)"%(ammo,ammo-pre_ammo), " HEALTH: %d (%d)"%(health,health-pre_health))
                    print("POSX:%.4f (%.4f) POSY: %.4f (%.4f)" % (posx,posx-pre_posx,posy, posy-pre_posy))
                    # print_variables(game)
                    
                    store_data(transit,step,s1,s2,idx,isterminal,health,ammo,frag,death-death_bias,posx, posy)

                    # tran = transition.Transition(s1,idx,s2,r,isterminal,True)
                    # tran.record_log(frag-pre_frag,death-pre_death,health-pre_health,ammo-pre_ammo,posx-pre_posx,posy-pre_posy)
                    # transitionset[step] = tran

                    step += 1
                else:
                    pass
            else:
                store_data(transit,step,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
                step += 1
        
        # replaymemory.store(np.copy(transitionset))

    if use_hdf5==True:
        hdf = h5py.File(demo_path+outputfile,'r+')
        dir = "demodata_"+str(2)
        hdf.create_group(dir)
        hdf.create_dataset(dir+"/state1",data=state1)
        hdf.create_dataset(dir+"/state2",data=state2)
        hdf.create_dataset(dir+"/actions",data=actions)
        hdf.create_dataset(dir+"/isterminals",data=isterminals)
        hdf.create_dataset(dir+"/healths",data=healths)
        hdf.create_dataset(dir+"/ammos",data=ammos)
        hdf.create_dataset(dir+"/frags",data=frags)
        hdf.create_dataset(dir+"/deaths",data=deaths)
        hdf.create_dataset(dir+"/posxs",data=posxs)
        hdf.create_dataset(dir+"/posys",data=posys)

        hdf.flush()
        hdf.close()
        print("SAVED")
    else:
        np.save(demo_path+"state1.npy",state1)
        np.save(demo_path+"state2.npy",state2)
        np.save(demo_path+"actions.npy",actions)
        np.save(demo_path+"isterminals.npy",isterminals)
        np.save(demo_path+"healths.npy",healths)
        np.save(demo_path+"ammos.npy",ammos)
        np.save(demo_path+"frags.npy",frags)
        np.save(demo_path+"deaths.npy",deaths)
        np.save(demo_path+"posxs.npy",posxs)
        np.save(demo_path+"posys.npy",posys)

    #total_reward = np.array(total_reward)
    #print("Results: mean: %.1f(+-)%.1f," % (total_reward.mean(), total_reward.std()), \
                  #"min: %.1f," % total_reward.min(), "max: %.1f," % total_reward.max())


    # replaymemory.save_data()
    print("FIN")
    game.close()
