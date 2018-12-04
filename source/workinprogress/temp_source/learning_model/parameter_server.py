class ParameterServer:
    def __init__(self, sess, log_dir):
        self.sess = sess
        with tf.variable_scope("parameter_server", reuse=tf.AUTO_REUSE):
            self.state1_ = tf.placeholder(tf.float32, shape=(None,) + RESOLUTION)
            self._build_model(self.state1_)

            self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="parameter_server")
            self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, RMSProbDecaly)
            
        with tf.variable_scope("summary", reuse=tf.AUTO_REUSE):
            self._build_summary(sess,log_dir)
        
        self.saver = tf.train.Saver()
        
#         print("-------GLOBAL-------")
#         for w in self.weights_params:
#             print(w)

    def _build_model(self,state):
            self.conv1 = NetworkSetting.conv1(state)
            self.maxpool1 = NetworkSetting.maxpool1(self.conv1)
            self.conv2 = NetworkSetting.conv2(self.maxpool1)
            self.maxpool2 = NetworkSetting.maxpool2(self.conv2)
            reshape = NetworkSetting.reshape(self.maxpool2)
            fc1 = NetworkSetting.fc1(reshape)

            with tf.variable_scope("policy"):
                self.policy = NetworkSetting.policy(fc1)
            
            with tf.variable_scope("value"):
                self.value = NetworkSetting.value(fc1)
                
            print("---------MODEL SHAPE-------------")
            print(state.get_shape())
            print(self.conv1.get_shape())
            print(self.conv2.get_shape())
            print(reshape.get_shape())
            print(fc1.get_shape())
            print(self.policy.get_shape())
            print(self.value.get_shape())
                
    def _build_summary(self,sess, log_dir):
        
        self.reward_ = tf.placeholder(tf.float32,shape=(), name="reward")
        self.frag_ = tf.placeholder(tf.float32, shape=(), name="frag")
        self.death_ = tf.placeholder(tf.float32, shape=(), name="death")
        self.loss_p_ = tf.placeholder(tf.float32, shape=(), name="loss_policy")
        self.loss_v_ = tf.placeholder(tf.float32, shape=(), name="loss_value")
        
        with tf.variable_scope("Summary_Score"):
            s = [tf.summary.scalar('reward', self.reward_, family="score"), tf.summary.scalar('frag', self.frag_, family="score"), tf.summary.scalar("death", self.death_, family="score")]
            self.summary_reward = tf.summary.merge(s)
        
        with tf.variable_scope("Summary_Loss"):
            self.summary_loss = tf.summary.merge([tf.summary.scalar('loss_policy', self.loss_p_, family="loss"), tf.summary.scalar('loss_value', self.loss_v_, family="loss")])
        
        with tf.variable_scope("Summary_Images"):
            conv1_display = tf.reshape(tf.transpose(self.conv1, [0,3,1,2]), (-1, self.conv1.get_shape()[1],self.conv1.get_shape()[2]))
            conv2_display = tf.reshape(tf.transpose(self.conv2, [0,3,1,2]), (-1, self.conv2.get_shape()[1],self.conv2.get_shape()[2]))
            conv1_display = tf.expand_dims(conv1_display, -1)
            conv2_display = tf.expand_dims(conv2_display, -1)

            state_shape = self.state1_.get_shape()
            conv1_shape = conv1_display.get_shape()
            conv2_shape = conv2_display.get_shape()

            s_img = []
            s_img.append(tf.summary.image('state',tf.reshape(self.state1_,[-1, state_shape[1], state_shape[2], state_shape[3]]), 1, family="state1"))
            s_img.append(tf.summary.image('conv1',tf.reshape(self.conv1,[-1, conv1_shape[1], conv1_shape[2], 1]), family="conv1"))
            s_img.append(tf.summary.image('conv2',tf.reshape(self.conv2,[-1, conv2_shape[1], conv2_shape[2], 1]), family="conv2"))

            self.summary_image = tf.summary.merge(s_img)
            
        with tf.variable_scope("Summary_Weights"):
            s = [tf.summary.histogram(values=w, name=w.name, family="weights") for w in self.weights_params]
            self.summary_weights = tf.summary.merge(s)

        self.writer = tf.summary.FileWriter(log_dir)
        
    def write_graph(self, sess):
        self.writer.add_graph(sess.graph)
        
    def write_score(self,sess, step ,reward, frag, death):
        m = sess.run(self.summary_reward, feed_dict={self.reward_:reward, self.frag_:frag, self.death_:death})
        return self.writer.add_summary(m, step)
    
    def write_loss(self,sess, step, l_p, l_v):
        m = sess.run(self.summary_loss, feed_dict={self.loss_p_: l_p, self.loss_v_:l_v})
        return self.writer.add_summary(m, step)
    
    def write_img(self,sess, step, state):
        m = sess.run(self.summary_image, feed_dict={self.state1_: state})
        return self.writer.add_summary(m, step)
    
    def write_weights(self, sess, step):
        m = sess.run(self.summary_weights)
        return self.writer.add_summary(m, step)
        
    def load_model(self, sess, model_path):
        self.saver.restore(sess, model_path)
    
    def save_model(self, sess,  model_path, step):
        self.saver.save(sess, model_path, global_step = step)
        
    def load_cnnweights(self, sess, weights_path):
        assert len(weights_path) == 4
        cnn_weights = self.weights_params[:4]
        w_demo = [np.load(w_p) for w_p in weights_path]
        plh = [tf.placeholder(tf.float32, shape=w.shape) for w in w_demo]
        assign_op = [w.assign(p) for w, p in zip(cnn_weights, plh)]
        feed_dict = {p:w for w,p in zip(w_demo, plh)}
        sess.run(assign_op, feed_dict)