class network_disable(object):

    def __init__(self,session,resolution,n_action, learning_rate):

        self.resolution = resolution

        self.s1_ = tf.placeholder(tf.float32, [None] + [resolution[0],resolution[1]] + [3], name="State")
        self.a_ = tf.placeholder(tf.int32, [None], name="Action")
        self.target_q_ = tf.placeholder(tf.float32, [None, n_action], name="TargetQ")
        self.reward_ = tf.placeholder(tf.float32, [1], name="reward_mean")
        self.reward = tf.identity(self.reward_)

        with tf.name_scope("conv1"):
            self.conv1_weight = tf.Variable(tf.truncated_normal(shape=[7,7,3,32],stddev=0.1),name="weight")
            self.conv1_strides = [1,2,2,1]
            self.conv1_padding = "SAME"
            self.conv1_bias = tf.Variable(tf.constant(0.1, shape=[32]),name="bias")

            self.conv1 = tf.nn.conv2d(self.s1_,self.conv1_weight,self.conv1_strides,self.conv1_padding,name="conv")
            self.conv1_relu = tf.nn.relu(tf.nn.bias_add(self.conv1, self.conv1_bias),name="relu")
        
        with tf.name_scope("conv1_target"):
            self.conv1_weight_target = tf.Variable(tf.constant(0.0,shape=[7,7,3,8]), name='weight')
            self.conv1_bias_target = tf.Variable(tf.constant(0.0,shape=[8]), name='bias_target')

            self.conv1_target = tf.nn.conv2d(self.s1_,self.conv1_weight_target,self.conv1_strides,self.conv1_padding,name="conv1_target")
            self.conv1_relu_target = tf.nn.relu(tf.nn.bias_add(self.conv1_target, self.conv1_bias_target), name="relu_target")

        with tf.name_scope("conv2"):
            self.conv2_weight = tf.Variable(tf.truncated_normal(shape=[3,3,32,8],stddev=0.1),name="weight")
            self.conv2_strides = [1,2,2,1]
            self.conv2_padding = "SAME"
            self.conv2_bias = tf.Variable(tf.constant(0.1,shape=[8]))
            self.conv2 = tf.nn.conv2d(self.conv1_relu, self.conv2_weight, self.conv2_strides,self.conv2_padding,name="conv")
            self.conv2_relu = tf.nn.relu(tf.nn.bias_add(self.conv2,self.conv2_bias),name="relu")

        with tf.name_scope("maxpool1"):
            self.maxpool1_ksize = [1,3,3,1]
            self.maxpool1_stride = [1,2,2,1]
            self.maxpool1_padding = "SAME"

            self.maxpool1 = tf.nn.max_pool(self.conv2_relu, self.maxpool1_ksize, strides=self.maxpool1_stride, padding=self.maxpool1_padding,name="maxpool")

        with tf.name_scope("conv3"):
            self.conv3_weight = tf.Variable(tf.truncated_normal(shape=[3,3,8,128],stddev=0.1),name="weight")
            self.conv3_strides = [1,2,2,1]
            self.conv3_padding = "SAME"
            self.conv3_bias = tf.Variable(tf.constant(0.1,shape=[128]))
            self.conv3 = tf.nn.conv2d(self.maxpool1, self.conv3_weight, self.conv3_strides, self.conv3_padding,name="conv")
            self.conv3_relu = tf.nn.relu(tf.nn.bias_add(self.conv3, self.conv3_bias),name="relu")

        with tf.name_scope("maxpool2"):
            self.maxpool2_ksize = [1,3,3,1]
            self.maxpool2_stride = [1,2,2,1]
            self.maxpool2_padding = "SAME"
            self.maxpool2 = tf.nn.max_pool(self.conv3_relu, self.maxpool2_ksize,strides = self.maxpool2_stride, padding=self.maxpool2_padding,name="maxpool")

        with tf.name_scope("reshape"):
            self.reshape = tf.contrib.layers.flatten(self.maxpool2)

        with tf.name_scope("fc1"):
            self.fc1_weights = tf.Variable(tf.truncated_normal(shape=[self.reshape.get_shape()[1].value,128],stddev=0.1),name="weight")
            self.fc1_bias = tf.Variable(tf.constant(0,1,shape=[128]),name="bias")
            self.fc1 = tf.nn.bias_add(tf.matmul(self.reshape,self.fc1_weights),self.fc1_bias,name="fc")
            self.fc1_relu = tf.nn.relu(self.fc1,name="relu")

        with tf.name_scope("fc2"):
            self.fc2_weights = tf.Variable(tf.truncated_normal(shape=[128,n_action],stddev=0.1),name="weight")
            self.fc2_bias = tf.Variable(tf.constant(0,1,shape=[n_action]),name="bias")
            self.fc2 = tf.nn.bias_add(tf.matmul(self.fc1_relu, self.fc2_weights),self.fc2_bias,name="fc")
            self.q = tf.nn.relu(self.fc2,name="relu")

        s1_shape = self.s1_.get_shape()
        conv1_shape = self.conv1_relu.get_shape()
        conv2_shape = self.conv2_relu.get_shape()
        maxpool1_shape = self.maxpool1.get_shape()
        conv3_shape = self.conv3_relu.get_shape()
        maxpool2_shape = self.maxpool2.get_shape()
        reshape_shape = self.reshape.get_shape()
        fc1_shape = self.fc1.get_shape()
        q_shape = self.q.get_shape()

        print(s1_shape[1:])
        print(conv1_shape[1:])
        print(conv2_shape[1:])
        print(maxpool1_shape[1:])
        print(conv3_shape[1:])
        print(maxpool2_shape[1:])

        print(self.s1_.name,"-------")
        print(s1_shape,"\n")

        print(self.conv1_relu.name,"-------")
        print(conv1_shape,"\n")

        print(self.conv2_relu.name,"-------")
        print(conv2_shape,"\n")

        print(self.maxpool1.name,"-------")
        print(maxpool1_shape,"\n")

        print(self.conv3_relu.name,"-------")
        print(conv3_shape,"\n")

        print(self.maxpool2.name,"-------")
        print(maxpool2_shape,"\n")

        print(self.reshape.name,"-------")
        print(reshape_shape,"\n")

        print(self.fc1_relu.name,"-------")
        print(fc1_shape,"\n")

        print(self.q.name,"-------")
        print(q_shape,"\n")

        self.best_action = tf.argmax(self.q,1)

        self.loss = tf.losses.mean_squared_error(self.q,self.target_q_)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate)

        self.train_step = self.optimizer.minimize(self.loss)

        self.session = session

        self.init = tf.global_variables_initializer()
        self.session.run(self.init)

        self.saver = tf.train.Saver()

        with tf.name_scope("summary"):
            #tf.summary.image('s1_',tf.reshape(self.s1_,[-1]+list(s1_shape[1:])),1)
            #tf.summary.image('conv1',tf.reshape(tf.transpose(self.conv1_relu,perm=[0,3,1,2]),[-1]+list(conv1_shape[1:])),1)
            #tf.summary.image('conv2',tf.reshape(tf.transpose(self.conv2_relu,perm=[0,3,1,2]),[-1]+list(conv2_shape[1:])),1)
            #tf.summary.image('maxpool1',tf.reshape(tf.transpose(self.maxpool1,perm=[0,3,1,2]),[-1]+list(maxpool1_shape[1:])),1)
            #tf.summary.image('conv3',tf.reshape(tf.transpose(self.conv3_relu,perm=[0,3,1,2]),[-1]+list(conv3_shape[1:])),1)
            #tf.summary.image('maxpool2',tf.reshape(tf.transpose(self.maxpool2,perm=[0,3,1,2]),[-1]+list(maxpool2_shape[1:])),1)
            tf.summary.scalar('loss', self.loss)
            #tf.summary.scalar('reward_mean', self.reward_)
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter("./logs", self.session.graph)

    def learn(self, s1, target,reward, step):
        l, _, m = self.session.run([self.loss, self.train_step,self.merged], feed_dict={self.s1_:s1, self.target_q_:target, self.reward_:reward})
        if step %10 == 0:
            self.writer.add_summary(m,step)
        return l

    def get_q_values(self, state):
        return self.session.run(self.q, feed_dict={self.s1_:state})

    def get_q_values_target(self,state):
        return self.session.run(self.)

    def get_best_action(self,state):
        return self.session.run(self.best_action, feed_dict={self.s1_:state})

    def get_best_action_simple(self, state):
        return self.get_best_action(state.reshape([1,self.resolution[0],self.resolution[0],3]))

    def write(self, s1):
        self.writer.add_summary(self.session.run(self.merged,feed_dict={self.s1_:s1}))

    def save_model(self, model_path,id):
        self.saver.save(self.session, model_path)

    def restore_model(self,model_path):
        self.saver.restore(self.session,model_path, id)
    
    def weights_and_biases(self):
        return [self.conv1_weight, self.conv1_bias, 
                self.conv2_weight, self.conv2_bias,
                self.conv3_weight, self.conv3_bias,
                self.fc1_weights, self.fc1_bias,
                self.fc2_weights, self.fc2_bias]

    def copy_params(self,training_network):
        target_params = training_network.weights_and_biases()
        origin_params = self.weights_and_biases()
        self.copyop = [tf.assign(origin, target), for origin,target in zip(origin_params,target_params) ]
        self.session.run(self.copyop)
