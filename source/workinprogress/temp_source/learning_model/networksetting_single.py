class NetworkSetting:
    
    def conv1(pre_layer):
        num_outputs = 32
#         kernel_size = [1,6,6]
#         stride = [1,3,3]
        kernel_size = [6,6]
        stride = [3,3]
        padding = 'SAME'
        activation = tf.nn.relu
        weights_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(0.1)
        return tf.contrib.layers.conv2d(pre_layer,kernel_size=kernel_size,\
                                        num_outputs=num_outputs,\
                                        stride=stride,padding=padding,activation_fn=activation,\
                                        weights_initializer=weights_init,\
                                        biases_initializer=bias_init)
    
    def maxpool1(pre_layer):
        return tf.nn.max_pool(pre_layer,[1,3,3,1],[1,2,2,1],'SAME')
    
    def conv2(pre_layer):
        num_outputs = 32
#         kernel_size = [1,3,3]
#         stride = [1,2,2]
        kernel_size = [3,3]
        stride = [2,2]
        padding = 'SAME'
        activation = tf.nn.relu
        weights_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(0.1)
        return tf.contrib.layers.conv2d(pre_layer,kernel_size=kernel_size,num_outputs=num_outputs,\
                                        stride=stride,padding=padding,activation_fn=activation,\
                                        weights_initializer=weights_init,biases_initializer=bias_init)
    
    def maxpool2(pre_layer):
        return tf.nn.max_pool(pre_layer,[1,3,3,1],[1,2,2,1],'SAME')
        
    def reshape(pre_layer):
        shape = pre_layer.get_shape()
        return tf.reshape(pre_layer, shape=(-1, shape[1]*shape[2]*shape[3]))
    
    def rnn(pre_layer, state):
        batch_size = tf.shape(pre_layer)[0]
        temp = tf.reduce_max(state, axis=4)
        temp = tf.reduce_max(temp, axis=3)
        temp = tf.reduce_max(temp, axis=2)
        lengh = tf.cast(tf.reduce_sum(tf.sign(temp) , axis=1),dtype=tf.int32)
        lengh = tf.where(tf.equal(lengh, tf.zeros_like(lengh)), tf.ones_like(lengh), lengh)
        cell = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE)
        rnn_state = cell.zero_state(batch_size, dtype=tf.float32)
        rnn_out, state_out = tf.nn.dynamic_rnn(cell, pre_layer, initial_state=rnn_state, sequence_length=lengh,dtype=tf.float32)
        out_idx = tf.range(0, batch_size) * N_ADV + (lengh  -1)
        output = tf.gather(tf.reshape(rnn_out, [-1, LSTM_SIZE]), out_idx)
        return output, lengh, rnn_out
    
    def fc1(pre_layer):
        num_outputs =1024
        activation_fn = tf.nn.relu
        weights_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.1)
        return tf.contrib.layers.fully_connected(pre_layer,num_outputs=num_outputs,activation_fn=activation_fn,weights_initializer=weights_init, biases_initializer=bias_init)
    
    def policy(pre_layer):
        num_outputs=N_AGENT_ACTION
        activation_fn = tf.nn.softmax
        weights_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.1)
        return tf.contrib.layers.fully_connected(pre_layer,num_outputs=num_outputs,activation_fn=activation_fn,weights_initializer=weights_init, biases_initializer=bias_init)
    
    def value(pre_layer):
        num_outputs = 1
        activation_fn = None
        weights_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.1)
        
        return tf.contrib.layers.fully_connected(pre_layer,num_outputs=num_outputs,activation_fn=activation_fn,weights_initializer=weights_init, biases_initializer=bias_init)
