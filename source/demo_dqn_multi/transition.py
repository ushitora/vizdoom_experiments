
class Transition(object):

    def __init__(self,s1,action,s2,reward,isterminal,isdemo):
        
        self.s1 = s1
        self.s2 = s2
        self.action = action
        self.reward = reward
        self.isterminal = isterminal
        self.isdemo = isdemo

        self.m_health = 0.0
        self.m_death = 0.0
        self.m_frag = 0.0
        self.m_ammo = 0.0
        self.m_posx = 0.0
        self.m_posy = 10.0

    def record_log(self,m_frag=0.0,m_death=0.0,m_health=0.0,m_ammo=0.0,m_posx=0.0,m_posy=0.0):
        self.m_health = m_health
        self.m_frag=m_frag
        self.m_death = m_death
        self.m_ammo = m_ammo
        self.m_posx = m_posx
        self.m_posy = m_posy
