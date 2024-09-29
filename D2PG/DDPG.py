import tensorflow as tf
import numpy as np
from UAV_env import UAVEnv
import time
import matplotlib.pyplot as plt
from state_normalization import StateNormalization

#####################  hyper parameters  ####################

MAX_EPISODES = 1000
LR_A = 0.001  # Actor的学习率
LR_C = 0.002  # Critic的学习率
GAMMA = 0.999  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
VAR_MIN = 0.01  # exploration noise minimum variance
NOISE_DECAY = 0.995  # 噪声衰减率
OUTPUT_GRAPH = False

###############################  DDPG with Double Q Networks  ####################################

class DDPG:
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)  # 初始化Replay Buffer
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.S = tf.placeholder(tf.float32, [None, s_dim], 'state')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 'next_state')
        self.R = tf.placeholder(tf.float32, [None, 1], 'reward')
        # self.a = tf.placeholder(tf.float32, [None, a_dim], 'action')

        # 创建 Actor 和 Critic 网络
        with tf.variable_scope('Actor'):
            self.a_eval = self._build_actor(self.S, scope='eval', trainable=True)
            self.a_target = self._build_actor(self.S_, scope='target', trainable=False)

        with tf.variable_scope('Critic'):
            # Critic 1: 计算 Q 值，用于训练 Actor
            # self.q1_eval = self._build_critic(self.S, self.a_eval, scope='eval1', trainable=True)
            # self.q1_target = self._build_critic(self.S_, self.a_target, scope='target1', trainable=False)

            # Critic 2: 第二个 Critic 网络用于减少 Q 值的高估
            self.q2_eval = self._build_critic(self.S, self.a_eval, scope='eval2', trainable=True)
            self.q2_target = self._build_critic(self.S_, self.a_target, scope='target2', trainable=False)

        # 获取参数
        self.ae_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/eval')  # 确保使用 TRAINABLE_VARIABLES
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        # self.c1e_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic/eval1')
        # self.c1t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target1')
        self.c2e_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic/eval2')
        self.c2t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target2')

        # 软更新目标网络
        self.soft_replace = [
            [tf.assign(t, (1 - TAU) * t + TAU * e) for t, e in zip(self.at_params, self.ae_params)],
            # [tf.assign(t, (1 - TAU) * t + TAU * e) for t, e in zip(self.c1t_params, self.c1e_params)],
            [tf.assign(t, (1 - TAU) * t + TAU * e) for t, e in zip(self.c2t_params, self.c2e_params)]
        ]

        # Critic 目标：取 Critic 1 和 Critic 2 的最小值
        # q_target = self.R + GAMMA * tf.minimum(self.q1_target, self.q2_target)
        q_target = self.R + GAMMA * self.q2_target
        # Critic 网络损失函数
        # td_error1 = tf.losses.mean_squared_error(labels=q_target, predictions=self.q1_eval)
        td_error2 = tf.losses.mean_squared_error(labels=q_target, predictions=self.q2_eval)
        # critic_loss = td_error1 + td_error2  # 最小化两个 Critic 网络的误差
        critic_loss = td_error2
        # self.critic_train_op = tf.train.AdamOptimizer(LR_C).minimize(critic_loss, var_list=self.c1e_params + self.c2e_params)
        self.critic_train_op = tf.train.AdamOptimizer(LR_C).minimize(critic_loss, var_list=self.c2e_params)


        # Actor 网络损失函数：最大化 Critic Q1 值 (从 Critic1 得到的 Q 值反馈)
        actor_loss = -tf.reduce_mean(self.q2_eval)  # 使用 Critic 的输出作为 Actor 的优化目标
        self.actor_train_op = tf.train.AdamOptimizer(LR_A).minimize(actor_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

        if OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        return self.sess.run(self.a_eval, {self.S: state})[0]

    def learn(self):
        # 软更新目标网络
        self.sess.run(self.soft_replace)

        # 从 Replay Buffer 中随机采样
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        batch_memory = self.memory[indices, :]
        bs = batch_memory[:, :self.s_dim]
        ba = batch_memory[:, self.s_dim:self.s_dim + self.a_dim]
        br = batch_memory[:, -self.s_dim - 1: -self.s_dim]
        bs_ = batch_memory[:, -self.s_dim:]

        # 更新 Critic 网络
        self.sess.run(self.critic_train_op, {self.S: bs, self.a_eval: ba, self.R: br, self.S_: bs_})

        # 更新 Actor 网络
        self.sess.run(self.actor_train_op, {self.S: bs})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # 替换旧数据
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_actor(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 400, activation=tf.nn.relu6, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 100, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 20, activation=tf.nn.relu, name='l3', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound[1], name='scaled_a')

    def _build_critic(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 400
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 100, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 20, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


###############################  training  ####################################

np.random.seed(1)
tf.set_random_seed(1)

env = UAVEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound  # 动作空间的上下限 [-1,1]
MAX_EP_STEPS = env.slot_num
ddpg = DDPG(a_dim, s_dim, a_bound)

# var = 1.0  # 初始噪声
var = 0.005  # 初始噪声
decay_rate = NOISE_DECAY  # 噪声衰减
ep_reward_list = []
s_normal = StateNormalization()
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0

    j = 0
    while j < MAX_EP_STEPS:
        # Add exploration noise
        a = ddpg.choose_action(s_normal.state_normal(s))
        a = np.clip(np.random.normal(a, var), *a_bound)  # 高斯噪声add randomness to action selection for exploration
        s_, r, is_terminal, step_redo, offloading_ratio_change, reset_dist = env.step(a)
        if step_redo:
            continue
        if reset_dist:
            a[2] = -1
        if offloading_ratio_change:
            a[3] = -1
        ddpg.store_transition(s_normal.state_normal(s), a, r, s_normal.state_normal(s_))  # 训练奖励缩小10倍

        if ddpg.pointer > BATCH_SIZE:
            # var = max([var * 0.9997, VAR_MIN])  # decay the action randomness
            ddpg.learn()
        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS - 1 or is_terminal:
            print('Episode:', i, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward, 'Explore: %.3f' % var)
            ep_reward_list = np.append(ep_reward_list, ep_reward)
            # file_name = 'output_ddpg_' + str(self.bandwidth_nums) + 'MHz.txt'
            file_name = 'output.txt'
            with open(file_name, 'a') as file_obj:
                file_obj.write("\n======== This episode is done ========")  # 本episode结束
            break
        j = j + 1
    var = max([var * decay_rate, VAR_MIN])

    # # Evaluate episode
    # if (i + 1) % 50 == 0:
    #     eval_policy(ddpg, env)

print('Running time: ', time.time() - t1)
plt.plot(ep_reward_list)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
