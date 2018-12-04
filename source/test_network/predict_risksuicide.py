
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from vizdoom import *
import skimage.color, skimage.transform
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_fscore_support
from random import sample, randint, random
import time,random,threading,datetime
from tqdm import tqdm
import tensorflow as tf
import h5py
import sys, os, glob
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from game_instance import GameInstance
# from global_constants import *
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import  RandomOverSampler


# In[ ]:


__name__ = "learning"

DEMO_PATH = ["./demonstration/predict_risksuicide/data_risksuicide_simple%02d.hdf5"%(i) for i in [1,2]]
MODEL_DIR = "./models/predict_risksuicide/model_test/"
LOG_DIR = "./logs/log_predict_risksuicide/log_test"

LABELS = os.path.join(os.getcwd(), "./logs/log_predict_risksuicide/label_simple_256.tsv")
SPRITES = os.path.join(os.getcwd(), "./logs/log_predict_risksuicide/sprite_simple_256.png")
SPRITES_DATA = os.path.join(os.getcwd(),'./logs/log_predict_risksuicide/sprite_img_simple_256.npy')

RESOLUTION = (120, 120, 3)

THRESHOLD = 30
KEEP_PROB = 0.7
BATCH_SIZE = 64


# In[ ]:


for f in os.listdir(LOG_DIR):
       print("removed  ", f)
       os.remove(os.path.join(LOG_DIR, f))


# In[ ]:


def under_sampling(x,y):
    n_positive = sum(y)
    x_indice = np.array(range(len(x))).reshape((-1,1))
    rus = RandomUnderSampler(ratio={0:n_positive*3, 1:n_positive})
    x_indice_resampled, y_resampled = rus.fit_sample(x_indice,y)
    x_resampled = x[x_indice_resampled.reshape((-1,))]
    return x_resampled, y_resampled, x_indice_resampled.reshape((-1,))
#     return x_resampled, y_resampled

def over_sampling(x,y):
    n_positive = sum(y)
    n_negative = len(y) - n_positive
    x_indice = np.arange(0,len(x)).reshape((-1,1))
    ros = RandomOverSampler(ratio={0:n_negative, 1:n_negative})
    x_indice_resampled, y_resampled = ros.fit_sample(x_indice, y)
    x_resampled = x[x_indice_resampled.reshape((-1))]
    return x_resampled, y_resampled


# In[ ]:


def load_data():
    batch_img = []
    batch_label = []
    for d in DEMO_PATH:
        print("loading "+d)
        file = h5py.File(d, "r")
        episode_list = list(file.keys())[1:]

        for e in episode_list[:]:
            n_steps = file[e+"/states"].shape[0]
            states = file[e+"/states"][:]
            damages = file[e+"/damages"][:]

            for img in states:
                batch_img.append(img)

            for damage in damages:
                if damage > THRESHOLD:
                    batch_label.append(1)
                else:
                    batch_label.append(0)

        file.close()
    return np.array(batch_img), np.array(batch_label)


# In[ ]:


class NetworkLocal(object):
    def __init__(self,name):
        self.name = name
        
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.state1_ = tf.placeholder(tf.float32,shape=(None, )+RESOLUTION, name="state_1")
            self.target_ = tf.placeholder(tf.int32, shape=(None,), name="area")
            self.keep_prob_ = tf.placeholder(tf.float32, name = "keep_prob")
            self.q_model = self._model(self.state1_, self.keep_prob_)
            self._build_graph()
            self.saver = tf.train.Saver(self.weights_params)

#         print("-----LOCAL weights---")
#         for w in self.weights_params:
#             print(w)
            
#         print("-----LOCAL grads---")
#         for w in self.grads:
#             print(w)
    
    def _model(self,state, keep_prob):

        self.conv1 = NetworkSetting.conv1(state)
        maxpool1 = NetworkSetting.maxpool1(self.conv1)
        self.conv2 = NetworkSetting.conv2(maxpool1)
        maxpool2 = NetworkSetting.maxpool2(self.conv2)
        reshape = NetworkSetting.reshape(maxpool2)
#         rnn ,l ,_ = NetworkSetting.lstm(reshape, state)
        fc1 = NetworkSetting.fc1(reshape)
        drop = NetworkSetting.dropout(fc1, keep_prob)
        
        q_value = NetworkSetting.q_value(drop)
        
        return q_value

    def _build_graph(self):
        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        
        self.prob = tf.nn.softmax(self.q_model, axis=1)
        
        self.onehot = tf.one_hot(self.target_, depth=2)
        self.loss_batch = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.q_model, labels=self.onehot) + 1e-5 * tf.reduce_sum([tf.nn.l2_loss(w) for w in self.weights_params])
        self.loss = tf.reduce_mean(self.loss_batch)
        with tf.variable_scope("trainer"):
            optimizer = tf.train.AdamOptimizer()
            self.update_step = optimizer.minimize(self.loss)
            
        with tf.variable_scope("Grad-CAM"): 
            cost = self.loss_batch
#             cost = tf.reduce_sum((network.prob - onehot) ** 2)
#             cost = (-1) * tf.reduce_sum(tf.multiply(onehot, tf.log(network.prob)), axis=1)
            y_c = tf.reduce_sum(tf.multiply(self.onehot, self.q_model), axis=1)

            self.target_conv_layer_grad = tf.gradients(y_c, self.conv1)[0]
            self.gb_grad = tf.gradients(y_c, self.state1_)[0]
        return 0
    
    def update_parameter_server_batch(self, s1, target):

        weights = SESS.run(self.weights_params)
        assert np.isnan([np.mean(w) for w in weights]).any()==False , print(weights)
        feed_dict = {self.state1_: s1, self.target_:target, self.keep_prob_:KEEP_PROB}
        l,_ = SESS.run([self.loss, self.update_step], feed_dict)
        return l

    def predict_enemyposition(self, s1):
        
        if np.ndim(s1) == 3:
            s1 = np.array([s1])
            probs = SESS.run(self.prob, {self.state1_:s1, self.keep_prob_:1.0})
            return [np.random.choice(2, p=p) for p in probs][0]
        elif np.ndim(s1) == 4:
            probs = SESS.run(self.prob, {self.state1_:s1, self.keep_prob_:1.0})
            return [np.random.choice(2, p=p) for p in probs]
        else:
            return None

    def get_q_values(self, s1):
        if np.ndim(s1) == 3:
            s1 = np.array([s1])
            q = SESS.run(self.q_model, {self.state1_:s1, self.keep_prob_:1.0})
            return q[0]
        elif np.ndim(s1) == 4:
            q = SESS.run(self.q_model, {self.state1_:s1, self.keep_prob_:1.0})
            return q
        else:
            return None
        
    def get_probability(self, s1):
        if np.ndim(s1) == 3:
            s1 = np.array([s1])
            p = SESS.run(self.prob, {self.state1_:s1, self.keep_prob_:1.0})
            return p[0]
        elif np.ndim(s1) == 4:
            p = SESS.run(self.prob, {self.state1_:s1, self.keep_prob_:1.0})
            return p
        else:
            return None
        
    def get_loss(self, s1, target):
        if np.ndim(s1) == 3:
            s1 = np.array([s1])
            q = SESS.run(self.loss, {self.state1_:s1,  self.target_:target, self.keep_prob_:1.0})
            return q[0]
        elif np.ndim(s1) == 4:
            q = SESS.run(self.loss, {self.state1_:s1,  self.target_:target, self.keep_prob_:1.0})
            return q
        else:
            return None
        
    def get_gradient_image(self, img, target):
        return SESS.run([self.gb_grad, self.target_conv_layer_grad], {self.state1_:img, self.target_:target, self.keep_prob_:1.0})
        
    def get_score(self, s1, target):
        pred = self.predict_enemyposition(s1)
        return sum(pred==target) / len(target)
    
    def save_model(self, model_path):
        return self.saver.save(SESS, model_path)
    
    def load_model(self, model_path):
        return self.saver.restore(SESS, model_path)


# In[ ]:


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
        return tf.contrib.layers.conv2d(pre_layer,kernel_size=kernel_size,                                        num_outputs=num_outputs,                                        stride=stride,padding=padding,activation_fn=activation,                                        weights_initializer=weights_init,                                        biases_initializer=bias_init)
    
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
        return tf.contrib.layers.conv2d(pre_layer,kernel_size=kernel_size,num_outputs=num_outputs,                                        stride=stride,padding=padding,activation_fn=activation,                                        weights_initializer=weights_init,biases_initializer=bias_init)
    
    def maxpool2(pre_layer):
        return tf.nn.max_pool(pre_layer,[1,3,3,1],[1,2,2,1],'SAME')
        
    def reshape(pre_layer):
        shape = pre_layer.get_shape()
        return tf.reshape(pre_layer, shape=(-1, shape[1]*shape[2]*shape[3]))
    
    def fc1(pre_layer):
        num_outputs =1024
        activation_fn = tf.nn.relu
        weights_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.1)
        return tf.contrib.layers.fully_connected(pre_layer,num_outputs=num_outputs,activation_fn=activation_fn,                                                 weights_initializer=weights_init, biases_initializer=bias_init)
    
    def dropout(pre_layer, keep_prob):
        return tf.nn.dropout(pre_layer, keep_prob)
    
    def q_value(pre_layer):
        num_outputs =2
        activation_fn = None
        weights_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.1)
        return tf.contrib.layers.fully_connected(pre_layer,num_outputs=num_outputs,activation_fn=activation_fn,                                                 weights_initializer=weights_init, biases_initializer=bias_init)


# In[ ]:


class LogRecorder(object):
    def __init__(self,log_dir, sprites = None, labels=None):
        
        # Place holder
        self.state1_ = tf.placeholder(tf.float32,shape=(None,)+RESOLUTION, name="state1")
        with tf.variable_scope("log_recorder", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("model"):
                self.conv1, self.conv2,self.embedding_input, self.model = self._build_model(self.state1_)

            with tf.variable_scope("Summary_Images"):
                conv1_display = tf.reshape(tf.transpose(self.conv1, [0,3,1,2]), (-1, self.conv1.get_shape()[1],self.conv1.get_shape()[2]))
                conv2_display = tf.reshape(tf.transpose(self.conv2, [0,3,1,2]), (-1, self.conv2.get_shape()[1],self.conv2.get_shape()[2]))
                conv1_display = tf.expand_dims(conv1_display, -1)
                conv2_display = tf.expand_dims(conv2_display, -1)
                self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="log_recorder")
                

            with tf.variable_scope("Summary_Loss"):
                # Summary for LOSS
                self.loss_ = [tf.placeholder(tf.float32,shape=()), tf.placeholder(tf.float32, shape=())]
                loss_name = ["loss_test","loss_train" ]
                self.merged_loss = self._build_scalar_summary(self.loss_, loss_name, "loss")

            # Summary for  SCORE
            with tf.variable_scope("Summary_Score"):
                self.accuracy_ = [tf.placeholder(tf.float32, shape=()), tf.placeholder(tf.float32, shape=())]
                accuracy_name = ["accuracy_test", "accuracy_train"]
                self.merged_accuracy = self._build_scalar_summary(self.accuracy_, accuracy_name, "score")

            # Summary for SCREEN
            with tf.variable_scope("Summary_Images"):
                image_name = ["state1", "conv1", "conv2"] 
                self.merged_images = self._build_image_summary([self.state1_, conv1_display,conv2_display],n_output=10, names=image_name, family="states")

            with tf.variable_scope("Summary_Filter"):
                #Sumamry for FILTER
                for w in self.weights_params:
                    print(w.get_shape())
                display_filter_conv1 = tf.reshape(tf.transpose(self.weights_params[0], [2,3,0,1]), (-1,self.weights_params[0].get_shape()[0], self.weights_params[0].get_shape()[1]))
                display_filter_conv2 = tf.reshape(tf.transpose(self.weights_params[2], [2,3,0,1]), (-1,self.weights_params[2].get_shape()[0], self.weights_params[2].get_shape()[1]))
                display_filter_conv1 = tf.expand_dims(display_filter_conv1, -1)
                display_filter_conv2 = tf.expand_dims(display_filter_conv2, -1)
                print(display_filter_conv1.get_shape())
                print(display_filter_conv2.get_shape())
                filter_shapes = [display_filter_conv1.get_shape(), display_filter_conv2.get_shape()]
                filter_names = ["conv1", "conv2"]
                self.merged_filters = self._build_image_summary([display_filter_conv1, display_filter_conv2], 10, filter_names, "filters")

            with tf.variable_scope("Summary_Histogram"):
                # Summary WEIGHT HISTOGRAM
                self.merged_weight_histograms = self._build_weight_histogram(self.weights_params, "weights")

            with tf.variable_scope("Summary_Embedding"):
                # Embedding
                if sprites is not None and labels is not None:
                    self.embedding = tf.Variable(tf.zeros([256, self.embedding_input.get_shape()[1]]), name="test_embedding")
                    self.assignment = self.embedding.assign(self.embedding_input)
                    self.saver = tf.train.Saver([self.embedding])
                
            self.writer = tf.summary.FileWriter(log_dir)
            self.writer.add_graph(SESS.graph)
            
            if sprites is not None and labels is not None:
                conf = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
                embedding_config = conf.embeddings.add()
                embedding_config.tensor_name = self.embedding.name
                embedding_config.sprite.image_path = sprites
                embedding_config.metadata_path = labels
                embedding_config.sprite.single_image_dim.extend([120,120,3])
                tf.contrib.tensorboard.plugins.projector.visualize_embeddings(self.writer, conf)
    
    def _build_model(self,state):
        conv1 = NetworkSetting.conv1(state)
        maxpool1 = NetworkSetting.maxpool1(conv1)
        conv2 = NetworkSetting.conv2(maxpool1)
        maxpool2 = NetworkSetting.maxpool2(conv2)
        reshape = NetworkSetting.reshape(maxpool2)
        fc1 = NetworkSetting.fc1(reshape)
        
        q_value = NetworkSetting.q_value(fc1)
        
        return conv1, conv2,fc1,q_value
    
    def _build_scalar_summary(self, placeholders, names, family):
        return tf.summary.merge([tf.summary.scalar(n, i, family=family) for n,i in zip(names, placeholders)])
    
    def _build_image_summary(self, placeholders,n_output, names, family):
        summaries = []
        for i, p in enumerate(placeholders):
            shape = p.get_shape().as_list()
            summ = tf.summary.image(names[i], p, n_output, family=family)
            summaries.append(summ)
        return tf.summary.merge(summaries)
    
    def _build_weight_histogram(self, weights, family):
        print([w.name for w in weights])
        s = [tf.summary.histogram(values=w, name=w.name, family=family) for w in weights]
        return tf.summary.merge(s)
        
    def write_loss(self, step, loss_test, loss_train):
        m = SESS.run(self.merged_loss, {self.loss_[0]:loss_test, self.loss_[1]:loss_train})
        return self.writer.add_summary(m, step)
    
    def write_accuracy(self, step, acc_test, acc_train):
        m = SESS.run(self.merged_accuracy, {self.accuracy_[0]:acc_test, self.accuracy_[1]:acc_train})
        return self.writer.add_summary(m, step)
    
    def write_images(self, step, s1):
        feed_dict = {self.state1_:s1}
        m = SESS.run(self.merged_images, feed_dict)
        return self.writer.add_summary(m, step)
    
    def write_filters(self, step):
        m = SESS.run(self.merged_filters)
        return self.writer.add_summary(m, step)
    
    def write_weights(self, step):
        m = SESS.run(self.merged_weight_histograms)
        return self.writer.add_summary(m, step)
    
    def write_embedding(self, step, model_path,  img):
        SESS.run(self.assignment, feed_dict={self.state1_: img})
        return self.saver.save(SESS, model_path, step)
    
    def copy_weights(self, network):
        SESS.run([i.assign(j) for i,j in zip(self.weights_params, network.weights_params)])


# In[ ]:


if __name__ == "learning":
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list="0"))
    config.log_device_placement = False
    config.allow_soft_placement = True
    SESS = tf.Session(config=config)

    with tf.device("/gpu:0"):
        network = NetworkLocal("test")

    log_rec = LogRecorder(log_dir=LOG_DIR, sprites=SPRITES, labels=LABELS)


    images, labels = load_data()
#     label = np.array([l[-1] for l in label_row])
    labels = labels.astype(np.int32)
    
    images , labels, images_indices = under_sampling(images, labels)
    
    train_img, test_img , train_label, test_label = train_test_split(images, labels, train_size=0.8, random_state=1)

    n_train = np.shape(train_img)[0]
    n_test = np.shape(test_label)[0]
    print("n_train:",n_train, "n_positive:", sum(train_label))
    print("n_test:", n_test, "n_positive:", sum(test_label))

    SESS.run(tf.global_variables_initializer())
    
    TRAIN_ACC = []
    TRAIN_LOSS = []
    TEST_ACC = []
    TEST_LOSS = []
    
    sprite_img = np.load(SPRITES_DATA)
    
    for i in tqdm(range(2000)):
        batch_idx = np.random.randint(n_train, size=BATCH_SIZE)
        batch_img = train_img[batch_idx]
        batch_label = train_label[batch_idx]
        l = network.update_parameter_server_batch(batch_img, batch_label)
        if (i+1) % 10 == 0:
            log_rec.copy_weights(network)

            batch_idx = np.random.randint(n_train, size=50)
            train_acc = network.get_score(train_img[batch_idx], train_label[batch_idx])
            train_loss = network.get_loss(train_img[batch_idx], train_label[batch_idx])
            
            batch_idx = np.random.randint(n_test, size=50)
            test_loss = network.get_loss(test_img[batch_idx], test_label[batch_idx])
            test_acc = network.get_score(test_img[batch_idx], test_label[batch_idx])

            log_rec.write_loss(i,test_loss, train_loss)
            log_rec.write_accuracy(i,test_acc, train_acc)
        
            TRAIN_LOSS.append(train_loss)
            TRAIN_ACC.append(train_acc)
            TEST_ACC.append(test_acc)
            TEST_LOSS.append(test_loss)
            
            log_rec.write_weights(i)
            log_rec.write_filters(i)
            log_rec.write_images(i,images[100:120])
            
            if (i+1) % 100 == 0:
                log_rec.write_embedding(i, os.path.join(LOG_DIR, 'embedded.ckpt'), sprite_img)

            
    network.save_model(os.path.join(MODEL_DIR, 'model.ckpt'))


# In[ ]:


if __name__=="learning":
#     network.load_model(MODEL_PATH)
    predict_label = []
    for s in test_img:
        predict_label.append(network.predict_enemyposition([s])[0])

    print(n_test)
    confusion_mat = confusion_matrix(test_label, predict_label)
    print(confusion_mat)
    scores = precision_recall_fscore_support(test_label, predict_label)
    print("---PRECISION---\n",  scores[0])
    print("---RECALL---\n", scores[1])
    print("---FSCORE---\n", scores[2])
    print("---NUMBER of LABELS---\n", scores[3])


# In[ ]:


if __name__ == "learning":
    x = range(len(TEST_ACC))
    plt.plot(x, np.convolve(TEST_ACC, np.ones(5)/5,mode="same"), "r",label="Test Accuracy")
    plt.plot(x, np.convolve(TRAIN_ACC, np.ones(5)/5,mode="same"),"b", label="Train Accuracy")
    plt.legend()
    x = range(len(TRAIN_LOSS))
    plt.plot(x, np.convolve(TEST_LOSS, np.ones(5)/5,mode="same"), "r",label="Test Loss")
    plt.plot(x, np.convolve(TRAIN_LOSS, np.ones(5)/5,mode="same"),"b", label="Train Loss")
    plt.legend()


# In[ ]:


__name__="analysis"


# In[ ]:


if __name__ == "analysis":
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list="0"))
    config.log_device_placement = False
    config.allow_soft_placement = True
    SESS = tf.Session(config=config)

    with tf.device("/gpu:0"):
        network = NetworkLocal("test")
    network.load_model(os.path.join(MODEL_DIR, 'model.ckpt'))
    
    
    test_img = np.load(SPRITES_DATA)
    test_label = pd.read_csv(LABELS, delimiter="\t", header=None).values
    test_label = np.ravel(test_label)
    predict_label = network.predict_enemyposition(test_img)
    confusion_mat = confusion_matrix(test_label, predict_label)
    print(confusion_mat)
    
    print(test_img.shape)
    print(test_label.shape)
    
    gb_grad_values,  target_conv_layer_grad_values = network.get_gradient_image(test_img, test_label)


# In[ ]:


IDX = []
for i,l in enumerate(test_label):
    if(l==1):
        IDX.append(i)
idx = 0


# In[ ]:


print(test_label[IDX[idx]])
plt.subplot(1,3,1)
plt.imshow(gb_grad_values[IDX[idx]])
plt.subplot(1,3,2)
plt.imshow(target_conv_layer_grad_values[IDX[idx], :, :, 0])
plt.subplot(1,3,3)
plt.imshow(test_img[IDX[idx]])
idx += 1


# In[ ]:


if __name__=="make_sprite":
    
    idx_0 = np.where(test_label==0)[0][:128]
    idx_1 = np.where(test_label==1)[0][:128]

    idx = np.concatenate([idx_0, idx_1], axis=0)
    idx.sort()

    sprite_img = test_img[idx]
    sprite_label = test_label[idx]
    
    np.save( './logs/log_predict_risksuicide/sprite_img_simple_256.npy', sprite_img)

    sprite_img = np.reshape(sprite_img, (16,16,120,120,3))

    img_over = np.concatenate([ np.concatenate([sprite_img[i,j] for j in range(16)] , axis=1) for i in range(16)] , axis=0)
    img_over *= 255
    img_over = img_over.astype(np.int32)
    img_over = Image.fromarray(np.uint8(img_over))
    img_over.save('./logs/log_predict_risksuicide/sprite_simple_256.png')

    df = pd.DataFrame({'label':sprite_label})
    df.to_csv('./logs/log_predict_risksuicide/label_simple_256.tsv', sep='\t', index=False, header=False)


# In[ ]:


if __name__ == "makedata":
    def preprocess(img):
        if len(img.shape) == 3 and img.shape[0]==3:
            img = img.transpose(1,2,0)
        
        img = skimage.transform.resize(img, RESOLUTION, mode="constant")
        img = img.astype(np.float32)
        return img

    def convert_action_agent2engine(agent_action):
        assert type(agent_action) == type(int()) or type(agent_action) == type(np.int64()), print("type(agent_action)=",type(agent_action))
        ans = []
        for i in range(6):
            ans.append(agent_action%2)
            agent_action = int(agent_action / 2)
        return ans
    
    game = GameInstance(game=DoomGame(), config_file_path=CONFIG_FILE_PATH,name='test', n_adv=5, rewards=REWARDS)
    IMG_BUFF = []
    DAMAGE_BUFF = []
    ATTACK_BUFF = []
    pre_health = 100.0
    for i in range(10):
        game.new_episode(0)
        while not game.is_episode_finished():
            s = preprocess(game.get_screen_buff())
            h = game.get_health()
            engine_action =convert_action_agent2engine( np.random.randint(32) * 2)
            game.make_action(engine_action, 4)
            IMG_BUFF.append(s)
            DAMAGE_BUFF.append(pre_health - game.get_health())
            pre_health = game.get_health()

            if (game.is_player_dead()):
                game.respawn_player()
                pre_health = 100.0
                
    DAMAGE_BUFF = np.array(DAMAGE_BUFF)
    IMG_BUFF= np.array(IMG_BUFF)
    
    idx_negative_sample = np.where(DAMAGE_BUFF <= 0)[0]
    n_negative_sample = idx_negative_sample.shape[0]
    idx_positive_sample = np.where(DAMAGE_BUFF > 0)[0]
    
    deleted_idx = np.random.choice(idx_negative_sample, int(n_negative_sample/2))
    IMG_BUFF = np.delete(IMG_BUFF, deleted_idx, axis=0)
    DAMAGE_BUFF = np.delete(DAMAGE_BUFF, deleted_idx, axis=0)
    
    GROUP = "4"
    with h5py.File('data_risksuicide01.hdf5', "r+") as f:
        f.create_group(GROUP+"/")


        f.create_dataset(GROUP+"/states", data=IMG_BUFF)
        f.create_dataset(GROUP+"/damages", data=DAMAGE_BUFF)


# In[ ]:


if __name__ == "save_weights":
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list="0"))
    config.log_device_placement = False
    config.allow_soft_placement = True
    SESS = tf.Session(config=config)

    with tf.device("/gpu:0"):
        network = NetworkLocal("test")
    
    network.load_model(MODEL_DIR+"model.ckpt")
    weights = SESS.run(network.weights_params)
    for w,name in zip(weights, ["conv1_kernel.npy", "conv1_bias.npy", "conv2_kernel.npy", "conv2_bias.npy"]):
        np.save("./weights_suicide/"+name, w)

