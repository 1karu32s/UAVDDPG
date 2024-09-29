import tensorflow as tf
import numpy as np
from UAV_env import UAVEnv
import time
import matplotlib.pyplot as plt
from state_normalization import StateNormalization

#####################  hyper parameters  ####################
MAX_EPISODES = 1000
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.99  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
N_STEP_RETURN = 5  # N-step return for better exploration
N_ATOMS = 51  # Number of bins for the distribution of returns
V_MIN = -10  # Min value of return
V_MAX = 10  # Max value of return
DELTA_Z = (V_MAX - V_MIN) / (N_ATOMS - 1)  # Width of each atom

Z = np.linspace(V_MIN, V_MAX, N_ATOMS)  # Atom values for the distribution

# Soft replacement
def soft_update(target, source, tau):
    for t, s in zip(target, source):
        t.assign((1 - tau) * t + tau * s)

###############################  D4PG  ####################################
class D4PG(object):

    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)  # memory storage
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')  # current state
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')  # next state
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')  # reward
        self.T = tf.placeholder(tf.float32, [None, N_ATOMS], 't_distribution')  # Target distribution for critic

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)  # Actor network for choosing actions
            a_ = self._build_a(self.S_, scope='target', trainable=False)  # Target Actor network

        with tf.variable_scope('Critic'):
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)  # Critic network
            self.q_ = self._build_c(self.S_, a_, scope='target', trainable=False)  # Target Critic network

        # Actor and Critic parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # Critic loss using cross-entropy for distributional RL
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=q, labels=self.T)
        self.critic_loss = tf.reduce_mean(cross_entropy)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(self.critic_loss, var_list=self.ce_params)

        # Actor loss
        a_loss = - tf.reduce_mean(tf.reduce_sum(q * Z, axis=1))  # Mean Q value from distribution
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        # Calculate target distribution using next state
        q_next = self.sess.run(self.q_, {self.S_: bs_})
        target_distribution = self.compute_target_distribution(br, q_next)

        # Train actor and critic
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.T: target_distribution})
        self.sess.run(self.atrain, {self.S: bs})

    def compute_target_distribution(self, rewards, q_next):
        """
        Compute target distribution for the critic, following the Distributional Bellman update.
        """
        Tz = np.clip(rewards + GAMMA * Z, V_MIN, V_MAX)  # Compute the projected returns
        b = (Tz - V_MIN) / DELTA_Z
        l = np.floor(b).astype(int)
        u = np.ceil(b).astype(int)

        # Build the target distribution
        target_distribution = np.zeros((BATCH_SIZE, N_ATOMS))
        for i in range(BATCH_SIZE):
            for j in range(N_ATOMS):
                target_distribution[i, l[i, j]] += (u[i, j] - b[i, j]) * q_next[i, j]
                target_distribution[i, u[i, j]] += (b[i, j] - l[i, j]) * q_next[i, j]

        return target_distribution

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 400, activation=tf.nn.relu6, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound[1], name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(tf.concat([s, a], axis=1), 400, activation=tf.nn.relu6, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            q_distribution = tf.layers.dense(net, N_ATOMS, activation=None, name='q_distribution', trainable=trainable)
            return tf.nn.softmax(q_distribution)  # Return the distribution over the atoms


###############################  training  ####################################
np.random.seed(1)
tf.set_random_seed(1)

env = UAVEnv()
MAX_EP_STEPS = env.slot_num
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

ddpg = D4PG(a_dim, s_dim, a_bound)
var = 0.01  # exploration noise
t1 = time.time()
ep_reward_list = []
s_normal = StateNormalization()

for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    j = 0

    while j < MAX_EP_STEPS:
        a = ddpg.choose_action(s_normal.state_normal(s))
        a = np.clip(np.random.normal(a, var), *a_bound)
        s_, r, is_terminal, step_redo, offloading_ratio_change, reset_dist = env.step(a)
        if step_redo:
            continue
        if reset_dist:
            a[2] = -1
        if offloading_ratio_change:
            a[3] = -1
        ddpg.store_transition(s_normal.state_normal(s), a, r, s_normal.state_normal(s_))

        if ddpg.pointer > BATCH_SIZE:
            ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS - 1 or is_terminal:
            print(f'Episode: {i}, Steps: {j}, Reward: {ep_reward:.2f}')
            ep_reward_list.append(ep_reward)
            break
        j += 1

print('Running time: ', time.time() - t1)
plt.plot(ep_reward_list)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
