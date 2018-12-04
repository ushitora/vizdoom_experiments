CONFIG_FILE_PATH = "./config/custom_config.cfg"
DEMO_PATH = ["./demonstration/demodata0"+"%01d"%(i)+".hdf5" for i in  range(1,8)]

N_WORKERS = 10

USED_GPU = "0"

RESOLUTION = (120,120,3)

REWARDS = {'living':-0.01, 'health_loss':-1, 'medkit':50, 'ammo':0.0, 'frag':500, 'dist':3e-2, 'suicide':-500} 

FRAME_REPEAT = 5

TURN_LEFT = 0
TURN_RIGHT = 1
MOVE_RIGHT = 2
MOVE_FORWARD = 3
MOVE_LEFT = 4
ATTACK = 5

AGENT_ACTIONS = [[MOVE_LEFT], [MOVE_FORWARD,MOVE_LEFT],[MOVE_FORWARD], [MOVE_RIGHT, MOVE_FORWARD], \
                [MOVE_RIGHT], [TURN_LEFT], [TURN_RIGHT], [TURN_LEFT,MOVE_LEFT], [TURN_RIGHT,MOVE_LEFT], \
                 [TURN_LEFT,MOVE_RIGHT], [TURN_RIGHT,MOVE_RIGHT], \
                 [TURN_LEFT,MOVE_FORWARD], [TURN_RIGHT,MOVE_FORWARD], \
                 [ATTACK]]
AGENT_ACTIONS_NAME = ["Move Left","Move Left Forward","Move Forward" ,"Move Right Forward","Move Right","Turn Left", "Turn Right", \
                      "Move Left Turn Left","Move Left Turn Right", "Move Right Turn Left", 
                      "Move Right Turn Right", "Move Forward Turn Left", "Move Forward Turn Right", "Fire"]
#[# Move Left \
    # Move Left Forward, \
    #Move Forward , \
    # Move Right Forward, \
    # Move Right, \
    #Turn Left, # Turn Right, \
    # Move Left Turn Left, \
    # Move Left Turn Right \
    # Move Right Turn Left \
    # Move Right Turn Right \
    # Move Forward Turn Left \
    # Move Forward Turn Right, \
    # Fire]



N_ADV = 5

BATCH_SIZE = 64
N_ACTION = 6
N_AGENT_ACTION = 2 ** N_ACTION
N_ENGINE_ACTION = N_ACTION

FREQ_COPY = 100
FREQ_TEST = 50

GAMMA = 0.99

LEARNING_RATE = 0.03
RMSProbDecaly = 0.9

LAMBDA1 = 0.1
LAMBDA2 = 1.0
LAMBDA3 = 0.0
L_MIN = 0.8
LSTM_SIZE = 128

EPS_START = 0.5
EPS_END = 0.0
LINEAR_EPS_START = 0.1
LINEAR_EPS_END = 0.9

BETA_0 = 0.0

CAPACITY = 4000

TEST_INTERVAL = 1000
RECORD_INTERVAL =50

BOTS_NUM = 5

MERGIN_BASE = 0.8

EPS_MAX = 0.9
EPS_MIN = 0.5
RESOLUTION = (120,120,3)