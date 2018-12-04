#!/usr/bin/env python3

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy np

# Converts and down-samples the input image
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img

# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game

class q_function(chainer.Chain):
    def __init__(self, n_history=1, n_action=3):
        super().__init__(
            conv1 = L.Convolution2D(n_history,8,ksize=8, stride=4, nobias=False)
            conv2 = L.Convolution2D(8,8,ksize=3, stride=2)
            fc1 = L.linear(None,128)
            out = L.linear(128, n_actions, initialW=np.zeros((n_action,128)),dtype~np.float32)
        )
    
    def __call__(self, x, test=False):
        s = chainer.Varialbe(x)
        h1 = F.relu(self.conv1(s))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.fc1(h2))
        h4 = F.relu(self.out(h3))

        return chaiinerrl.action_value.DiscreteActionValue(h4)

class dqn_agent(object):
    def __init__(self,)

if __name__="__main__":

    config_file_path = ""

    resolution = (80,60)

    gamma = 0.95

    n_epoch = 20
    steps_per_epoch = 2000

    bots_num = 10
    map = 1

    game = initialize_vizdoom(config_file_path)

    n_action = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0,1], repeat=n_action)]

    q_func = q_function(resolution,n_action=3)

    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(q_func)

    def random_action():
        return choice(n_action)

    explorer = chainerrl.explorers.ConstantEpsilonGreedy(epsilon=0.3, random_action

    replay_buff = chainerrl.replay_buffer.ReplayBuffer(capacity=10**6)

    phi = lambda x: x.astype(np.float32, copy=False)

    agent = chainerrl.agents.DQN(q_func,optimizer,replay_buff,gamma,explorer,replay_start_size=500,update_frequency=1,target_update_frequency=100,phi=phi)

    for epoch in range(n_epoch):
        print("EPISODE %d \n------------" % (epoch+1))
        train_scores = []
        finished_episodes = 0

        print("Training.....")
        game.new_episode()

        for learning_step in tqdm(range(steps_per_epoch)):

            screen_buff = game.get_state

            action = agent.act_and_train()

            if game.is_episode_finished():
                score=game.get_total_reward()
                train_scores.append(score)
                game.new_episode()
                finished_episodes += 1
        
        print("%d training episodes played." % train_episodes_finished)

        train_scores = np.array(train_scores)
        print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())


game = vzd.DoomGame()
game.load_config("config/custom_config.cfg")

# Name your agent and select color
# colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
name = "RandomAgent"
color = 0
game.add_game_args("+name {} +colorset {}".format(name, color))
game.init()

# Three sample actions: turn left/right and shoot
actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


# Add bots
for i in range(bots_num):
    game.send_game_command("addbot")

# Play until the game (episode) is over.

while not game.is_episode_finished():

    if game.is_player_dead():
        # Use this to respawn immediately after death, new state will be available.
        game.respawn_player()

        # Or observe the game until automatic respawn.
        # game.advance_action();
        # continue;

    # Analyze the state ... or not
    s = game.get_state()

    # Make your action.
    game.make_action(choice(actions))

    # Log your frags every ~5 seconds
    if s.number % 175 == 0:
        print("Frags:", game.get_game_variable(vzd.GameVariable.FRAGCOUNT))

game.close()
