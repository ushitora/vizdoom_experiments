class NetworkLocal(object):
    def __init__(self,name, parameter_server):
        self.name = name
        
        with tf.variable_scope(self.name+"_train", reuse=tf.AUTO_REUSE):
            self.state1_ = tf.placeholder(tf.float32,shape=(None,)+RESOLUTION, name="state_1")
            self.a_ = tf.placeholder(tf.float32, shape=(None,), name="action")
            self.r_adv = tf.placeholder(tf.float32, shape=(None,), name="reward_advantage")
            self.isterminal_ = tf.placeholder(tf.float32, shape=(None,), name="isterminal")
            self.policy, self.value, self.conv1, self.conv2 = self._model(self.state1_)

            self._build_graph(parameter_server)
            
            self.global_weights_ = [tf.placeholder(tf.float32, w.get_shape()) for w in self.weights_params]
            self.assign_weights = [l_p.assign(g_p) for l_p,g_p in zip(self.weights_params,self.global_weights_)]

    def _model(self,state):

#         with tf.variable_scope(self.name + "_nottrainable"):
        conv1 = NetworkSetting.conv1(state)
        maxpool1 = NetworkSetting.maxpool1(conv1)
        conv2 = NetworkSetting.conv2(maxpool1)
        maxpool2 = NetworkSetting.maxpool2(conv2)
        reshape = NetworkSetting.reshape(maxpool2)
        
        with tf.variable_scope(self.name + "_trainable"):
#             rnn ,l ,_ = NetworkSetting.lstm(reshape, state)
            fc1 = NetworkSetting.fc1(reshape)

            policy = NetworkSetting.policy(fc1)
            value = NetworkSetting.value(fc1)
        
        return policy, value, conv1, conv2

    def _build_graph(self, parameter_server):

        one_hot = tf.one_hot(tf.cast(self.a_, tf.int32), depth=N_AGENT_ACTION)
        
        log_prob = tf.log(tf.reduce_sum(self.policy * one_hot, axis=1, keep_dims=True)+1e-10)
        advantage = tf.reshape(self.r_adv, [-1,1]) - self.value
        self.loss_policy = -log_prob * tf.stop_gradient(advantage)
        self.loss_value = tf.square(advantage)
        self.loss_total = tf.reduce_mean(self.loss_policy + self.loss_value)
        
        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+"_train/")
        
        self.grads = tf.gradients(self.loss_total ,self.weights_params)
        
        self.update_global_weight_params = \
            parameter_server.optimizer.apply_gradients(zip(self.grads, parameter_server.weights_params))
        
        self.pull_global_weight_params = [l_p.assign(g_p) for l_p,g_p in zip(self.weights_params,parameter_server.weights_params)]

        self.push_local_weight_params = [g_p.assign(l_p) for g_p,l_p in zip(parameter_server.weights_params,self.weights_params)]
        
    def pull_parameter_server(self, sess):
        sess.run(self.pull_global_weight_params)
    
    def push_parameter_server(self):
        sess.run(self.push_local_weight_params)
        
    def get_weights(self, sess):
        return sess.run(self.weights_params)
    
    def get_gradients(self,sess, s1, a, r, isterminal):
        assert np.ndim(s1) == 4
        
        feed_dict = {self.state1_: s1, self.a_:a, self.r_adv:r,  self.isterminal_:isterminal}
        return sess.run(self.grads, feed_dict)
    
    def update_parameter_server(self,sess, s1, a, r, isterminal):
        assert np.ndim(s1) == 4
        feed_dict = {self.state1_: s1,self.a_:a, self.r_adv:r}
        _, l_p, l_v = sess.run([self.update_global_weight_params, self.loss_policy, self.loss_value],feed_dict)
        return l_p, l_v
    
    def check_weights(self, sess):
        weights = SESS.run(self.weights_params)
        assert np.isnan([np.mean(w) for w in weights]).any()==False , print(weights)

    def get_policy(self, sess, s1):
        assert np.ndim(s1) == 4    
        return sess.run(self.policy, {self.state1_: s1})

    def get_value(self, sess, s1):
        assert np.ndim(s1) == 4        
        return sess.run(self.value, {self.state1_:s1})