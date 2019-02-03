from vizdoom import *
import math


class GameInstanceSimpleDeathmatch(object):
    def __init__(self, game, name="Default", config_path=None, steps_update_origin=None, n_bots=None, reward_param=None, timelimit=2):
        self.name = name

        self.frag_count = 0
        self.death_count = 0
        self.kill_count = 0
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

        self.game = self.initialize_game(game, config_path)
        self.cross_x = self.game.get_screen_width()/2

    def initialize_game(self, game, config_file_path):
        game.load_config(config_file_path)
        game.set_window_visible(False)
        game.set_mode(Mode.PLAYER)
        game.set_screen_format(ScreenFormat.CRCGCB)
        game.set_screen_resolution(ScreenResolution.RES_640X480)
        game.set_labels_buffer_enabled(True)
        game.set_render_crosshair(True)
        game.set_render_weapon(True)
        game.set_episode_timeout(300)
        color = 4
        game.init()
        return game

    def new_episode(self):
        self.game.new_episode()
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
        self.ammo = self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        self.damage_count = 0
        self.posx = self.game.get_game_variable(GameVariable.POSITION_X)
        self.posy = self.game.get_game_variable(GameVariable.POSITION_Y)

        self.origin_x = self.posx
        self.origin_y = self.posy

    def update_variables(self,step):
        new_frag_count = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        new_death_count = self.game.get_game_variable(GameVariable.DEATHCOUNT)
        new_kill_count = self.game.get_game_variable(GameVariable.KILLCOUNT)
        new_ammo = self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        new_health = self.game.get_game_variable(GameVariable.HEALTH)
        new_posx = self.game.get_game_variable(GameVariable.POSITION_X)
        new_posy = self.game.get_game_variable(GameVariable.POSITION_Y)
        new_damage_count = self.game.get_game_variable(GameVariable.DAMAGECOUNT)
        
        enemy_insight = self.is_enemy()

        reward, reward_detail = self.get_reward(new_frag_count - self.frag_count, new_death_count - self.death_count,new_kill_count-self.kill_count, \
                            new_health-self.health, new_ammo-self.ammo, new_posx-self.origin_x, new_posy-self.origin_y,enemy_insight)

        self.frag_count = new_frag_count
        self.death_count = new_death_count
        self.kill_count = new_kill_count
        self.ammo = new_ammo
        self.health = new_health
        self.posx = new_posx
        self.posy = new_posy
        self.damage_count = new_damage_count

        if step % self.n_adv == 0:
            self.origin_x = self.posx
            self.origin_y = self.posy

        return reward, reward_detail

    def make_action(self, step ,action,framerepeat):
        self.game.make_action(action,framerepeat)       
        reward, reward_detail = self.update_variables(step)
        return reward, reward_detail
    
    def advance_action(self, step,framerepeat):
        self.game.advance_action(framerepeat)
        reward, reward_detail = self.update_variables(step)
        return reward, reward_detail

    def get_reward(self, m_frag,m_death, m_kill, m_health, m_ammo, m_posx, m_posy, enemysight):
        reward_detail = {'living':0.0,'frag':0.0, 'suicide':0.0,'kill':0.0, 'dist':0.0,'medkit':0.0, 'healthloss':0.0,'ammo':0.0,'death':0.0, 'enemysight':0.0, 'ammoloss':0.0}

        reward_detail['living'] = self.reward_param['living']

        if m_frag > 0:
            reward_detail['frag'] = (m_frag) * self.reward_param['frag']
            reward_detail['suicide'] = 0.0
        else:
            reward_detail['suicide'] = (m_frag*-1) * self.reward_param['suicide']
            reward_detail['frag'] = 0.0
            
        if m_death > 0:
            reward_detail['death'] = (m_death) * self.reward_param['death']
        
        if m_kill > 0:
            reward_detail['kill'] = self.reward_param['kill']
        else:
            reward_detail['kill'] = 0.0
            
        if enemysight == True: 
            reward_detail['enemysight'] = self.reward_param['enemysight']
            
        if m_ammo < 0:
            reward_detail['ammoloss'] = self.reward_param['ammoloss']

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
            if(l.object_id != 0 and l.object_name=="Cacodemon"):
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
        if self.game.get_state() is not None:
            for l in self.game.get_state().labels:
                if(l.object_id != 0 and l.object_name=="Cacodemon"):
                    return True
        else:
            return True

        return False
    
    def is_aiming(self):
        l = self.get_enemy_label()
        if self.cross_x > l[0] and self.cross_x < l[0] + l[2]:
            return True
        else:
            return False
    

class GameInstanceBasic(object):
    def __init__(self, game, name="Default", config_path=None, steps_update_origin=None, n_bots=None, reward_param=None, timelimit=2):
        self.name = name

        self.frag_count = 0
        self.death_count = 0
        self.kill_count = 0
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

        self.game = self.initialize_game(game, config_path)
        self.cross_x = self.game.get_screen_width()/2

    def initialize_game(self, game, config_file_path):
        game.load_config(config_file_path)
        game.set_window_visible(False)
        game.set_mode(Mode.PLAYER)
        game.set_screen_format(ScreenFormat.CRCGCB)
        game.set_screen_resolution(ScreenResolution.RES_640X480)
        game.set_labels_buffer_enabled(True)
        game.set_render_crosshair(False)
        game.set_render_weapon(False)
        game.init()
        return game

    def new_episode(self):
        self.game.new_episode()
        return 0

    def is_episode_finished(self):
        return self.game.is_episode_finished()
    
    def is_player_dead(self):
        return self.game.is_player_dead()
    
    def respawn_player(self):
        self.game.respawn_player()
        return 0

    def make_action(self, step ,action,framerepeat):
        r = self.game.make_action(action,framerepeat)
        new_killcount = self.get_kill_count()
        if new_killcount - self.kill_count > 0:
            r = self.reward_param['kill']
            r_detail = {'living':0.0, 'kill':self.reward_param['kill']}
        else:
            r = self.reward_param['living']
            r_detail = {'living':self.reward_param['living'], 'kill':0.0}
            
        self.kill_count = new_killcount
        return r, r_detail
    
    def advance_action(self, step,framerepeat):
        self.game.advance_action(framerepeat)
        reward, reward_detail = self.update_variables(step)
        return reward, reward_detail

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
            if(l.object_id != 0 and l.object_name=="Cacodaemon"):
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
        if self.is_episode_finished():
            return True
        else:
            for l in self.game.get_state().labels:
                if(l.object_id != 0 and l.object_name=="Cacodaemon"):
                    return True

        return False