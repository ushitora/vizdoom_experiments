from vizdoom import  *
import math

class GameInstanceCIG2017(object):
    def __init__(self, game, name="Default", config_path=None, steps_update_origin=None, n_bots=None, reward_param=None, timelimit=2):
        self.name = name
        
        self.frag_count = 0
        self.death_count = 0
        self.health = 100
        self.ammo = 15
        self.posx = 0.0
        self.poxy = 0.0
        self.hitcount = 0

        self.origin_x = 0.0
        self.origin_y = 0.0

        self.adv = 0
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
        game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_render_weapon(False)
        game.set_render_crosshair(False)
        game.add_game_args("-host 1 -deathmatch +timelimit {} ".format(self.timelimit)+"sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_nocrouch 1 "+"+viz_respawn_delay 10 + viz_nocheat 0")
        color = 0
        game.add_game_args("+name {} +colorset {}".format(self.name, color))
        game.init()
        return game

    def new_episode(self):
        self.game.new_episode()
        for i in range(self.n_bots):
            self.game.send_game_command("addbot")
        self.frag_count = 0
        self.kill_count = 0
        self.init_variables()
        return 0

    def is_episode_finished(self):
        return self.game.is_episode_finished()
    
    def is_player_dead(self):
        return self.game.is_player_dead()
    
    def respawn_player(self):
        self.game.respawn_player()
        for i in range(5):
            self.game.make_action([0,0,0,0,0,0], 5)
            self.init_variables()
        return 0

    def init_variables(self):
        self.health = 100
        self.ammo = 15
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
        new_hitcount = self.game.get_game_variable(GameVariable.HITCOUNT)
        
        if new_health < -1000:
            new_health = 100

        reward,reward_detail = self.get_reward(new_frag_count - self.frag_count, new_death_count - self.death_count, \
                            new_health-self.health, new_ammo-self.ammo, new_posx-self.origin_x, new_posy-self.origin_y, new_hitcount-self.hitcount)

        self.frag_count = new_frag_count
        self.death_count += new_death_count - self.death_count
        self.ammo = new_ammo
        self.health = new_health
        self.posx = new_posx
        self.posy = new_posy
        self.hitcount = new_hitcount

        if step == 0:
            self.origin_x = self.posx
            self.origin_y = self.posy

        return reward, reward_detail

    def make_action(self ,action, frame_repeat):
        self.game.make_action(action, frame_repeat)
        self.adv = (self.adv + 1) % self.n_adv
        return self.update_variables(self.adv)
    
    def advance_action(self, framerepeat):
        return self.game.advance_action(framerepeat)

    def get_reward(self, m_frag, m_death, m_health, m_ammo, m_posx, m_posy, m_hitcount):
        
        ret_detail = {}

        ret_detail['living'] = self.reward_param['living']

        if m_frag >= 0:
            ret_detail['frag'] = (m_frag)*self.reward_param['frag']
            ret_detail['suicide'] = 0.0
        else:
            ret_detail['suicide'] = (m_frag*-1)*(self.reward_param['suicide'])
            ret_detail['frag'] = 0.0
        
#         ret_detail['dist'] = int((math.sqrt((m_posx)**2 + (m_posy)**2))/self.dist_unit) * (self.rewards['dist'] * self.dist_unit)
        ret_detail['dist'] = (math.sqrt((m_posx)**2 + (m_posy)**2)) * self.reward_param['dist']
#         print("dist:", int((math.sqrt((m_posx)**2 + (m_posy)**2))/self.dist_unit))
#         print("dist:", math.sqrt((m_posx)**2 + (m_posy)**2))
#         print()
        
        if m_health > 0:
            ret_detail['medkit'] = self.reward_param['medkit']
            ret_detail['healthloss'] = 0.0
        else:
            ret_detail['medkit'] = 0.0
            ret_detail['healthloss'] = (m_health)*self.reward_param['healthloss'] * (-1)
        
        if m_hitcount > 0:
            ret_detail['hitcount'] = self.reward_param['hitcount'] * m_hitcount
        else:
            ret_detail['hitcount'] = 0.0

        ret_detail['ammo'] = (m_ammo)*self.reward_param['ammo'] if m_ammo>0 else 0.0
        
        return sum(ret_detail.values()), ret_detail 
    
    def get_screen_buff(self):
        return self.game.get_state().screen_buffer

    def get_available_buttons_size(self):
        return self.game.get_available_buttons_size()

    def get_frag_count(self):
        return self.frag_count

    def get_death_count(self):
        return self.death_count
    
    def get_kill_count(self):
        return self.game.get_game_variable(GameVariable.KILLCOUNT)

    def get_ammo(self):
        return self.ammo

    def get_health(self):
        return self.health

    def get_pos_x(self):
        return self.posx

    def get_pos_y(self):
        return self.posy

    def get_last_action(self):
        return self.game.get_last_action()

    def close(self):
        return self.game.close()

