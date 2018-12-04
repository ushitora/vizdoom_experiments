from vizdoom import *
import math

class reward_generater(object):
    def __init__(self,game):
        self.pre_health = game.get_game_variable(GameVariable.HEALTH)
        self.pre_ammo = game.get_game_variable(GameVariable.AMMO0)
        self.pre_frag = 0
        self.pre_pos_x = 0
        self.pre_pos_y = 0
        self.reward = 0
        self.total_reward = 0
        self.game = game

        # Reward
        self.rewards = {'living':-0.008, 'health_loss':-1, 'medkit':50, 'ammo':25, 'frag':100, 'dist':0.1}
    
    def update_reward(self):
        health = self.game.get_game_variable(GameVariable.HEALTH)
        ammo = self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        frag = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        pos_x = self.game.get_game_variable(GameVariable.POSITION_X)
        pos_y = self.game.get_game_variable(GameVariable.POSITION_Y)

        if self.pre_pos_x == 0 and self.pre_pos_y == 0:
            self.pre_pos_x = pos_x
            self.pre_pos_y = pos_y
        
        self.reward = 0.0

        self.reward += (frag-self.pre_frag)*self.rewards['frag']
        self.reward += (math.sqrt((pos_x-self.pre_pos_x)**2 + (pos_y-self.pre_pos_x)**2)) * self.rewards['dist']
        if health > self.pre_health:
            self.reward += self.rewards['medkit']
        elif self.pre_health < health:
            self.reward += (self.pre_health-health)*self.rewards['health_loss']
        else:
            pass
        
        self.total_reward += self.reward

        self.pre_frag = frag
        self.pre_health = health
        self.pre_ammo = ammo
        self.pre_pos_x = pos_x
        self.pre_pos_y = pos_y

    def update_reward_basic(self,reward):
        self.reward = reward
        self.total_reward += reward

    def reset_position(self):
        self.pre_pos_x = 0
        self.pre_pos_y = 0

    def reset_params(self):
        self.reset_position()
        self.pre_health = game.get_game_variable(GameVariable.HEALTH)
        self.pre_ammo = game.get_game_variable(GameVariable.AMMO0)
        self.pre_frag = 0
        self.reward = 0
        self.total_reward = 0

    def get_reward(self):
        return self.reward

    def get_total_reward(self):
        return self.total_reward