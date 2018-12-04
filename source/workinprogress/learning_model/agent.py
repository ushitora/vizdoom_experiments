#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class Agent(object):
    
    def __init__(self, network):
        self.network = network
        
        self.obs = {}
        self.obs['s1'] = np.zeros((N_ADV, )+ RESOLUTION, dtype=np.float32)
        
        self.states_buff = np.zeros((N_ADV*3-2,) + RESOLUTION, dtype=np.float32)
        self.rewards_buff = np.zeros((N_ADV*2-1,), dtype=np.float32)
        self.actions_buff = np.zeros((N_ADV*2-1, ), dtype=np.float32)
        self.isterminals_buff = np.ones((N_ADV*2-1, ), dtype=np.float32)
        self.buff_pointer = 0
        
    def calc_eps(self, progress):
        if progress < 0.2:
            return EPS_MIN
        elif progress >= 0.2 and progress < 0.8:
            return ((EPS_MAX - EPS_MIN)/ 0.6) * progress + ( EPS_MIN -  (EPS_MAX - EPS_MIN)/ 0.6 * 0.2)
        else :
            return EPS_MAX

    def act_eps_greedy(self, sess, s1, progress):
        assert progress >= 0.0 and progress <=1.0
        
        self.push_obs(s1)
        eps = self.calc_eps(progress)
        if random.random() <= eps:
            p = self.network.get_policy(sess, [self.obs['s1']])
            a_idx = np.random.choice(N_AGENT_ACTION, p=p)
        else:
            a_idx = np.random.randint(N_AGENT_ACTION)
            
        return a_idx
    
    def act_greedy(self, sess, s1):
        p = self.network.get_policy(sess, [self.obs['s1']])[0]
        a_idx = np.random.choice(N_AGENT_ACTION, p=p)
        return a_idx
    
    def get_gradients(self, sess):
        return self.network.get_gradients(sess, self.batch['s1'], self.batch['action'], self.batch['reward'], self.batch['isterminal'])
    
    def train_network(self, sess):
        batch={'s1':[], 'actions':[], 'rewards':[], 'isterminals':[]}
        for i in range(N_ADV):
            batch['s1'].append(self.states_buff[i:i+N_ADV])
            batch['actions'].append(self.actions_buff[i])
            batch['isterminals'].append(self.isterminals_buff[i])
            batch['rewards'].append(sum([r*GAMMA**j for j,r in self.rewards_buff[i:i+N_ADV]]))
        return self.network.train_network(sess, batch['s1'], batch['actions'], batch['rewards'], batch['isterminals'])
    
    def push_obs(self, s1):
        self.obs['s1'] = np.roll(self.obs['s1'],shift=-1, axis=0)
        self.obs['s1'][-1] = s1
        
    def clear_obs(self):
        self.obs = {}
        self.obs['s1'] = np.zeros((N_ADV,)+ RESOLUTION, dtype=np.float32)
        
    def push_batch(self, s1, action, reward, isterminal):
        self.states_buff = np.roll(self.states_buff, shift=-1, axis=0)
        self.actions_buff = np.roll(self.actions_buff, shift=-1, axis=0)
        self.rewards_buff = np.roll(self.rewards_buff, shift=-1, axis=0)
        self.isterminals_buff = np.roll(self.isterminals_buff, shift=-1, axis=0)
        self.states_buff[-1] = s1
        self.actions_buff[-1] = action
        self.rewards_buff[-1] = reward
        self.isterminals_buff[-1] = isterminal
        self.buff_pointer += 1
    
    def clear_batch(self):
        self.states_buff = np.zeros((N_ADV*3-2,) + RESOLUTION, dtype=np.float32)
        self.rewards_buff = np.zeros((N_ADV*2-1,), dtype=np.float32)
        self.actions_buff = np.zeros((N_ADV*2-1, ), dtype=np.float32)
        self.isterminals_buff = np.ones((N_ADV*2-1, ), dtype=np.float32)
        self.buff_pointer = 0


# In[ ]:




