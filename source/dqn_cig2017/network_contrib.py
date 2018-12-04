import tensorflow as tf

class network_contrib(object):

    def __init__(self,session,resolution,n_action, learning_rate):

        self.resolution = resolution

        self.s1_ = tf.placeholder(tf.float32, [None] + [resolution[0],resolution[1], resolution[2]], name="State")
        self.a_ = tf.placeholder(tf.int32, [None], name="Action")
        self.target_q_ = tf.placeholder(tf.float32, [None, n_action], name="TargetQ")
        self.reward_in = tf.placeholder(tf.float32, name="reward")
        self.reward_out = tf.identity(self.reward_in)

        self.conv1 = tf.contrib.layers.convolution2d(self.s1_, num_outputs=8, kernel_size=[6, 6], stride=[3, 3],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
        self.conv2 = tf.contrib.layers.convolution2d(self.conv1, num_outputs=8, kernel_size=[3, 3], stride=[2, 2],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
        self.conv2_flat = tf.contrib.layers.flatten(self.conv2)
        self.fc1 = tf.contrib.layers.fully_connected(self.conv2_flat, num_outputs=128, activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))

        self.q = tf.contrib.layers.fully_connected(self.fc1, num_outputs=n_action, activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.1))

        s1_shape = self.s1_.get_shape()
        conv1_shape = self.conv1.get_shape()
        conv2_shape = self.conv2.get_shape()
        conv2flat_shape = self.conv2_flat.get_shape()
        fc1_shape = self.fc1.get_shape()
        q_shape = self.q.get_shape()

        print(self.conv1.name,"-------")
        print(conv1_shape,"\n")

        print(self.conv2.name,"-------")
        print(conv2_shape,"\n")

        print(self.fc1.name,"-------")
        print(fc1_shape,"\n")

        print(self.q.name,"-------")
        print(q_shape,"\n")

        self.best_action = tf.argmax(self.q,1)

        self.loss = tf.losses.mean_squared_error(self.q,self.target_q_)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate)

        self.train_step = self.optimizer.minimize(self.loss)

        self.session = session

        #self.init = tf.global_variables_initializer()
        #self.session.run(self.init)

        #self.copy_params()

        #self.saver = tf.train.Saver()

        with tf.name_scope("summary"):
            tf.summary.image('s1_',tf.reshape(self.s1_,[-1]+list(s1_shape[1:])),1)
            #tf.summary.image('conv1',tf.reshape(self.conv1_relu, [-1]+list(conv1_shape[1:2])+[1]), 1)
            #tf.summary.image('conv2',tf.reshape(self.conv2_relu, [-1]+list(conv2_shape[1:2])+[1]), 1)
            tf.summary.image('conv1',tf.reshape(tf.transpose(self.conv1,perm=[0,3,1,2]),[-1]+list(conv1_shape[1:3])+[1]),1)
            tf.summary.image('conv2',tf.reshape(tf.transpose(self.conv2,perm=[0,3,1,2]),[-1]+list(conv2_shape[1:3])+[1]),1)
            #tf.summary.image('maxpool1',tf.reshape(tf.transpose(self.maxpool1,perm=[0,3,1,2]),[-1]+list(maxpool1_shape[1:])),1)
            #tf.summary.image('conv3',tf.reshape(tf.transpose(self.conv3_relu,perm=[0,3,1,2]),[-1]+list(conv3_shape[1:])),1)
            #tf.summary.image('maxpool2',tf.reshape(tf.transpose(self.maxpool2,perm=[0,3,1,2]),[-1]+list(maxpool2_shape[1:])),1)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('reward', self.reward_out)
            #tf.summary.scalar('reward_mean', self.reward_)
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter("./logs", self.session.graph)

    def learn(self, s1, target, reward, step):
        l, _, m = self.session.run([self.loss, self.train_step,self.merged], feed_dict={self.s1_:s1, self.target_q_:target, self.reward_in:reward})
        if step %10 == 0:
            self.writer.add_summary(m,step)
        return l

    def get_q_values(self, state):
        return self.session.run(self.q, feed_dict={self.s1_:state})

    def get_q_target_values(self,state):
        return self.session.run(self.q_target, feed_dict={self.s1_:state})

    def get_best_action(self,state):
        s1 = state.reshape([1,self.resolution[0],self.resolution[1],1])
        return self.session.run(self.best_action, feed_dict={self.s1_:s1})[0]

    def write(self, s1):
        self.writer.add_summary(self.session.run(self.merged,feed_dict={self.s1_:s1}))

"""
    def save_model(self, model_path,id):
        self.saver.save(self.session, model_path)

    def restore_model(self,model_path):
        self.saver.restore(self.session,model_path)
    def copy_params(self):
        origin_params = [self.conv1_weight, self.conv1_bias,self.conv2_weight, self.conv2_bias, self.fc1_weights,self.fc1_bias,self.fc2_weights,self.fc2_bias]
        target_params = [self.conv1_weight_target, self.conv1_bias_target, self.conv2_weight_target, self.conv2_bias_target, self.fc1_weight_target, self.fc1_bias_target, self.fc2_weight_target, self.fc2_bias_target]

        self.copyop = [tf.assign(origin, target) for origin,target in zip(origin_params,target_params) ]
        self.session.run(self.copyop)
"""

