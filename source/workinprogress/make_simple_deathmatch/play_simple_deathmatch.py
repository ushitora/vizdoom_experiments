#!/usr/bin/env python
from vizdoom import *
import numpy as np
import pandas as pd
import h5py, math
import skimage.transform
import sys, os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

CONFIG_PATH = "./config/simple_deathmatch.cfg"
RESOLUTION = (120,120)
CROSS = (200,320)
# SAVE_DEMO = True
SAVE_DEMO = False

REWARDS = { 'healthloss':-0.008, 'medkit':1, 'ammo':0.0, 'frag':1, 'dist':1e-3, 'suicide':-1}

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

class GameInstance(object):
    def __init__(self, game, name="Default", config_path=None, steps_update_origin=None, n_bots=None, reward_param=None, timelimit=2):
        self.name = name
        self.game = self.initialize_game(game, config_path)

        self.frag_count = 0
        self.death_count = 0
        self.health = 100
        self.ammo = 50
        self.posx = 0.0
        self.poxy = 0.0

        self.origin_x = 0.0
        self.origin_y = 0.0

        self.n_bots=n_bots
        self.n_adv = steps_update_origin
        self.reward_param = reward_param
        self.timelimit = timelimit

    def initialize_game(self, game, config_file_path):
        game.load_config(config_file_path)
        game.set_window_visible(True)
        game.set_mode(Mode.ASYNC_SPECTATOR)
        game.set_screen_format(ScreenFormat.CRCGCB)
        game.set_screen_resolution(ScreenResolution.RES_640X480)
        game.set_labels_buffer_enabled(True)
        game.set_render_crosshair(False)
        game.set_render_weapon(False)
        color = 4
        game.add_game_args("+name {} +colorset {}".format(self.name, color))
        game.add_game_args("-host 1 -deathmatch +viz_nocheat 0 +viz_debug 0 +viz_respawn_delay 10 +viz_nocheat 0 +timelimit %f"%(float(TIMELIMIT)))
        game.add_game_args("+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_crouch 1")
        game.init()
        print(self.name + " initialized.")
        return game

    def new_episode(self):
        self.game.send_game_command("removebots")
        for i in range(self.n_bots):
            self.game.send_game_command("addbot")

        # self.game.new_episode()

        self.init_variables()        
        return 0

    def is_episode_finished(self):
        return self.game.is_episode_finished()
    
    def is_player_dead(self):
        return self.game.is_player_dead()
    
    def respawn_player(self):
        self.game.respawn_player()
        self.init_variables()
        return 0

    def init_variables(self):
        self.frag_count = 0
        self.death_count = 0
        self.health = 100
        self.ammo = 15
        self.damage_count = 0
        self.posx = self.game.get_game_variable(GameVariable.POSITION_X)
        self.posy = self.game.get_game_variable(GameVariable.POSITION_Y)

        self.origin_x = self.posx
        self.origin_y = self.posy

    def update_variables(self,step):
        new_frag_count = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        new_death_count = self.game.get_game_variable(GameVariable.DEATHCOUNT)
        new_ammo = self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        new_health = self.game.get_game_variable(GameVariable.HEALTH)
        new_posx = self.game.get_game_variable(GameVariable.POSITION_X)
        new_posy = self.game.get_game_variable(GameVariable.POSITION_Y)
        new_damage_count = self.game.get_game_variable(GameVariable.DAMAGECOUNT)

        reward, reward_detail = self.get_reward(new_frag_count - self.frag_count, new_death_count - self.death_count, \
                            new_health-self.health, new_ammo-self.ammo, new_posx-self.origin_x, new_posy-self.origin_y)

        self.frag_count = new_frag_count
        self.death_count = new_death_count
        self.ammo = new_ammo
        self.health = new_health
        self.posx = new_posx
        self.posy = new_posy
        self.damage_count = new_damage_count

        if step % self.n_adv == 0:
            self.origin_x = self.posx
            self.origin_y = self.posy

        return reward, reward_detail

    def make_action(self, step ,action):
        self.game.make_action(action)
        return self.update_variables(step)
    
    def advance_action(self, framerepeat):
        return self.game.advance_action(framerepeat)

    def get_reward(self, m_frag, m_death, m_health, m_ammo, m_posx, m_posy):
        reward_detail = {}

        if m_frag > 0:
            reward_detail['frag'] = (m_frag) * self.reward_param['frag']
            reward_detail['suicide'] = 0.0
        else:
            reward_detail['suicide'] = (m_frag*-1) * self.reward_param['suicide']
            reward_detail['frag'] = 0.0

        reward_detail['dist'] = (math.sqrt((m_posx)**2 + (m_posy)**2)) * self.reward_param['dist']

        if m_health > 0:
            reward_detail['medkit'] = self.reward_param['medkit']
            reward_detail['healthloss'] = 0.0
        else:
            reward_detail['healthloss'] = m_health * self.reward_param['healthloss'] * (-1)
            reward_detail['medkit'] = 0.0

        if m_ammo > 0:
            reward_detail['ammo'] = (m_ammo) * self.reward_param['ammo']
        else:
            reward_detail['ammo'] = 0.0

        return sum(reward_detail.values()), reward_detail

    def get_screen_buff(self):
        return self.game.get_state().screen_buffer

    def get_available_buttons_size(self):
        return self.game.get_available_buttons_size()

    def get_frag_count(self):
        return self.game.get_game_variable(GameVariable.FRAGCOUNT)

    def get_death_count(self):
        return self.game.get_game_variable(GameVariable.DEATHCOUNT)

    def get_kill_count(self):
        return self.game.get_game_variable(GameVariable.KILLCOUNT)

    def get_ammo(self):
        return self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)

    def get_health(self):
        h = self.game.get_game_variable(GameVariable.HEALTH)
        if h < -1000:
            return 100

        return h

    def get_pos_x(self):
        return self.game.get_game_variable(GameVariable.POSITION_X)

    def get_pos_y(self):
        return self.game.get_game_variable(GameVariable.POSITION_Y)

    def get_hitcount(self):
        return self.game.get_game_variable(GameVariable.HITCOUNT)

    def get_damagecount(self):
        return self.damage_count
    def get_last_action(self):
        return self.game.get_last_action()

    def close(self):
        return self.game.close()

    def get_enemy_label(self):
        area = 0
        ans = [0,0,0,0]
        for l in self.game.get_state().labels:
            if(l.object_id != 0 and l.object_name=="DoomPlayer"):
                if area < l.width*l.height:
                    area = l.width*l.height
                    ans = [l.x, l.y, l.width, l.height]
        return ans

    def get_enemy_x(self):
        l = self.get_enemy_label()
        return l[0]

    def get_enemy_y(self):
        l = self.get_enemy_label()
        return l[1]

    def get_enemy_w(self):
        l = self.get_enemy_label()
        return l[2]

    def get_enemy_h(self):
        l = self.get_enemy_label()
        return l[3]

    def is_enemy(self):
        for l in self.game.get_state().labels:
            if(l.object_id != 0 and l.object_name=="DoomPlayer"):
                return True

        return False

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

    TIMELIMIT = args.timelimit

    os.system('clear')
    game_engine = DoomGame()
    game = GameInstance(game_engine, name="PLAYER1",config_path=CONFIG_PATH,steps_update_origin=10, n_bots=1, reward_param=REWARDS, timelimit=TIMELIMIT)

    # game.game.set_render_weapon(False)
    # game.game.set_render_crosshair(False)
    # game.game.init()
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

    game.new_episode()
    while not game.is_episode_finished():

        s_row = game.get_screen_buff()
        s,s_center = preprocess(s_row)
        label_enemy = [game.get_enemy_x(), game.get_enemy_y(), game.get_enemy_w(), game.get_enemy_h()]
        # s = game.get_screen_buff()
        game.advance_action(4)
        r,reward_detail =  game.update_variables(step)
        action = game.get_last_action()
        cur_frag_count = game.game.get_game_variable(GameVariable.FRAGCOUNT)

        pre_frag_count = cur_frag_count
        if REWARD_SAVE==True and not game.is_episode_finished():
            rewards_total['healthloss'].append(reward_detail['healthloss'])
            rewards_total['frag'].append(reward_detail['frag'])
            rewards_total['suicide'].append(reward_detail['suicide'])
            rewards_total['dist'].append(reward_detail['dist'])
            rewards_total['total'].append(r)

        if not action.count(1.0) == 0:
            if not game.is_episode_finished():
                sys.stdout.write("\r---%d----\n"%(step))
                sys.stdout.write("\rENEMY_POS: (%.2f, %.2f)\n"%(game.get_enemy_x(), game.get_enemy_y()))
                sys.stdout.write("\rENEMY_W_H: (%.2f, %.2f)\n"%(game.get_enemy_w(), game.get_enemy_h()))
                sys.stdout.write("\rREWARD:%.2f\n"%r)
                sys.stdout.write("\rREWARD_DETAIL:"+str(reward_detail)+"\n")
                sys.stdout.write("\rACTION:"+str(action)+"\n")
                sys.stdout.write("\rFRAG:%d DEATH:%d KILL:%d \n"%(game.get_frag_count(), game.get_death_count(), game.get_kill_count()))
                sys.stdout.write("\rHEALTH: %d AMMO:%d\n"%(game.get_health(), game.get_ammo()))
                sys.stdout.write("\rPOSX: %.3f POSY: %.3f\n"%(game.get_pos_x(), game.get_pos_y()))
                sys.stdout.flush()
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

            if game.is_player_dead():
                game.respawn_player()
        else:
            pass

    # print(tabulate([(int(game.game.get_game_variable(GameVariable.FRAGCOUNT)), int(game.game.get_game_variable(GameVariable.DEATHCOUNT)))], ['KILL', 'DEATH'], tablefmt='grid'))
    # print()

    if REWARD_SAVE == True:
        df_reward = pd.DataFrame(rewards_total)
        df_reward.to_csv(REWARDS_PATH)

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
    game.close()