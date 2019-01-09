#!/usr/bin/env python
from vizdoom import *
import numpy as np
import pandas as pd
import h5py
import skimage.transform
import sys, os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# from game_instance_simpledeathmatch import GameInstanceSimpleDeathmatch
from game_instance_basic import GameInstanceBasic

# CONFIG_PATH = "./config/simple_deathmatch_h80_s80.cfg"
CONFIG_PATH = "./config/large_basic.cfg"
RESOLUTION = (120,120)
CROSS = (200,320)
# SAVE_DEMO = True
SAVE_DEMO = False

REWARDS = {'frag':1, 'suicide':-1}

def preprocess(img):
    if len(img.shape) == 3:
        img = img.transpose(1,2,0)
    
    img_over = skimage.transform.resize(img, RESOLUTION)
    img_over = img_over.astype(np.float32)
    row_left = int(CROSS[0] - RESOLUTION[0]/2)
    row_right = row_left + RESOLUTION[0]
    col_left = int(CROSS[1] - RESOLUTION[1]/ 2)
    col_right = col_left + RESOLUTION[1]
    img_center = img[row_left:row_right, col_left:col_right, :]
    img_center = img_center.astype(np.float32) / 255.0
    return img_over, img_center

if __name__ == "__main__":

    parser = ArgumentParser("PLAY SCRIPT in Simple Deathmatch",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--demodata', metavar="DAMODATA_SAVE", dest='demodata_save',
                        default="", type=str,
                        help='path hdf5 file to save demonstration')
    parser.add_argument('-r', '--recordreward', metavar="REWARD_SAVE", dest='reward_save',
                        default="", type=str,
                        help='path csv file to save rewards')
    parser.add_argument('-t', '--time', metavar="TIMELIMIT", dest='timelimit',
                        default=2, type=int,
                        help='time limit for a match')
    parser.add_argument('-c', '--config', metavar="CONFIG_PATH", dest="config_path", default="", type=str)

    args = parser.parse_args()

    DEMO_PATH = args.demodata_save
    if DEMO_PATH == "":
        SAVE_DEMO = False
    else:
        if os.path.exists(DEMO_PATH):
            SAVE_DEMO = True
        else:
            exit()
    REWARDS_PATH = args.reward_save
    if REWARDS_PATH == "":
        REWARD_SAVE = False
    else:
        if os.path.exists(REWARDS_PATH):
            REWARD_SAVE = True
        else:
            exit()
    
    CONFIG_PATH = args.config_path

    TIMELIMIT = args.timelimit

    os.system('clear')
    game_engine = DoomGame()
    game = GameInstanceBasic(game_engine, name="PLAYER1",config_path=CONFIG_PATH,steps_update_origin=10, n_bots=1, reward_param=REWARDS, timelimit=TIMELIMIT)
    log_state = []
    log_state_center = []
    log_action = []
    log_frag = []
    log_death = []
    log_health = []
    log_ammo = []
    log_posx = []
    log_posy = []
    log_enemy_x = []
    log_enemy_y = []
    log_enemy_w = []
    log_enemy_h = []
    log_hitcount = []
    log_damagecount = []

    step = 0
    pre_time = 0
    pre_frag_count = 0

    rewards_total = {'healthloss':[], 'frag':[], 'dist':[], 'suicide':[], 'total':[]}
    game.n_bots = 0

    game.new_episode()
    while not game.is_episode_finished():

        s_row = game.get_screen_buff()
        s,s_center = preprocess(s_row)
        label_enemy = [game.get_enemy_x(), game.get_enemy_y(), game.get_enemy_w(), game.get_enemy_h()]
        reward,reward_detail = game.advance_action(step,4)
        action = game.get_last_action()
        print(action)

        if not action.count(1.0) == 0:
            step += 1
            if SAVE_DEMO == True and not game.is_episode_finished():
                log_state.append(s)
                log_state_center.append(s_center)
                log_action.append(action)
                log_frag.append(game.get_frag_count())
                log_death.append(game.get_death_count())
                log_health.append(game.get_health())
                log_ammo.append(game.get_ammo())
                log_posx.append(game.get_pos_x())
                log_posy.append(game.get_pos_y())
                log_enemy_x.append(label_enemy[0])
                log_enemy_y.append(label_enemy[1])
                log_enemy_w.append(label_enemy[2])
                log_enemy_h.append(label_enemy[3])
                log_hitcount.append(game.get_hitcount())
                log_damagecount.append(game.get_damagecount())
            elif SAVE_DEMO == True and game.is_episode_finished():
                    log_state.append(s)
                    log_state_center.append(s_center)
                    log_action.append(action)
                    log_frag.append(log_frag[-1])
                    log_death.append(log_death[-1])
                    log_health.append(log_health[-1])
                    log_ammo.append(log_ammo[-1])
                    log_posx.append(log_posx[-1])
                    log_posy.append(log_posy[-1])
                    log_enemy_x.append(label_enemy[0])
                    log_enemy_y.append(label_enemy[1])
                    log_enemy_w.append(label_enemy[2])
                    log_enemy_h.append(label_enemy[3])
                    log_hitcount.append(log_hitcount[-1])
                    log_damagecount.append(log_damagecount[-1])
            else:
                pass

        #print(reward_detail)
        print('%5.2f %5.2f %5.2f %5.2f %5.2f '%(reward,game.get_frag_count(), game.get_death_count(), game.get_pos_x(), game.get_pos_y()))
    game.close()

    if SAVE_DEMO == True:
      
        with h5py.File(DEMO_PATH, "r+") as f:
            groups = list(f.keys())
            groups = [int(s) for s in groups]
            groups.sort()
            print(groups)
            print("n_step:",len(log_action))
            new_group = str(groups[-1] + 1)
            print("No.",new_group,"'s data is saving")
            f.create_dataset(new_group+"/states", data=log_state)
            f.create_dataset(new_group+"/states_center", data=log_state_center)
            f.create_dataset(new_group+"/action", data=log_action)
            f.create_dataset(new_group+"/frag",data=log_frag)
            f.create_dataset(new_group+"/death",data=log_death)
            f.create_dataset(new_group+"/health",data=log_health)
            f.create_dataset(new_group+"/ammo",data=log_ammo)
            f.create_dataset(new_group+"/posx",data=log_posx)
            f.create_dataset(new_group+"/posy",data=log_posy)
            f.create_dataset(new_group+"/enemy_x",data=log_enemy_x)
            f.create_dataset(new_group+"/enemy_y", data=log_enemy_y)
            f.create_dataset(new_group+"/enemy_w", data=log_enemy_w)
            f.create_dataset(new_group+"/enemy_h", data=log_enemy_h)
            f.create_dataset(new_group+"/hitcount", data=log_hitcount)
            f.create_dataset(new_group+"/damagecount", data=log_damagecount)
