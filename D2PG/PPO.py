import tensorflow as tf
import numpy as np
from UAV_env import UAVEnv
import time
import random
import matplotlib.pyplot as plt
from state_normalization import StateNormalization

#####################  hyper parameters  ####################
MAX_EPISODES = 1000
GAMMA = 0.99  # reward discount
BATCH_SIZE = 64
LR_A = 0.0003  # learning rate for actor
LR_C = 0.0003  # learning rate for critic
CLIP_EPSILON = 0.2  # clipping range for PPO
UPDATE_STEPS = 10  # number of update steps
GAMMA = 0.99  # discount factor
LAMBDA = 0.95  # GAE lambda


class PPO(object):
    def __init__(self, s_dim, a_dim, a_bound, M):
        self.s_dim, self.a_dim, self.a_bound = s_dim, a_dim, a_bound
        self.M = M  # 保存UE的数量

        self.S = tf.placeholder(tf.float32, [None, s_dim], 'state')
        self.A = tf.placeholder(tf.float32, [None, a_dim], 'action')
        self.R = tf.placeholder(tf.float32, [None, 1], 'reward')
        self.Adv = tf.placeholder(tf.float32, [None, 1], 'advantage')

        with tf.variable_scope('policy'):
            self.pi, pi_params = self._build_a(self.S, scope='pi', trainable=True)  # 当前策略
            self.old_pi, oldpi_params = self._build_a(self.S, scope='old_pi', trainable=False)  # 旧策略

        with tf.variable_scope('critic'):
            self.v = self._build_c(self.S)  # 价值函数

        # PPO的策略损失
        ratio = tf.exp(
            tf.log(tf.reduce_sum(self.pi * self.A, axis=1)) - tf.log(tf.reduce_sum(self.old_pi * self.A, axis=1)))
        ratio = tf.expand_dims(ratio, axis=1)  # 手动扩展维度
        surr1 = ratio * self.Adv
        surr2 = tf.clip_by_value(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * self.Adv
        self.a_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))  # PPO裁剪目标

        # 价值函数损失
        self.c_loss = tf.reduce_mean(tf.square(self.R - self.v))

        # 优化器
        self.atrain_op = tf.train.AdamOptimizer(LR_A).minimize(self.a_loss)
        self.ctrain_op = tf.train.AdamOptimizer(LR_C).minimize(self.c_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(s, 100, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, 100, tf.nn.relu, trainable=trainable)
            action_prob = tf.layers.dense(l2, self.a_dim, tf.nn.softmax, trainable=trainable)
            return action_prob, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    def _build_c(self, s):
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(s, 100, tf.nn.relu)
            l2 = tf.layers.dense(l1, 100, tf.nn.relu)
            value = tf.layers.dense(l2, 1)
            return value

    def update(self, s, a, r, adv):
        # 更新策略和价值网络
        for _ in range(UPDATE_STEPS):
            self.sess.run(self.atrain_op, {self.S: s, self.A: a, self.Adv: adv})
            self.sess.run(self.ctrain_op, {self.S: s, self.R: r})

    def choose_action(self, s):
        # 获取策略网络的动作概率分布
        prob_weights = self.sess.run(self.pi, {self.S: s[np.newaxis, :]})

        # 从动作概率分布中选择动作
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())

        # 确保 action[0] 是 UE id，范围在 [0, M-1]，其余为连续变量
        ue_id = np.clip(action, 0, self.M - 1)  # 确保 action[0] 合法
        theta = random.uniform(-1, 1)  # action[1]: 飞行角度
        distance = random.uniform(-1, 1)  # action[2]: 飞行距离
        offloading_ratio = random.uniform(0, 1)  # action[3]: 卸载率

        # 返回多维动作
        return np.array([ue_id, theta, distance, offloading_ratio])

    def get_value(self, s):
        return self.sess.run(self.v, {self.S: s[np.newaxis, :]})[0]


###############################  training  ####################################
np.random.seed(2)
tf.set_random_seed(2)

env = UAVEnv()
env = UAVEnv()  # 假设环境有定义
M = env.M  # 从环境中获取 UE 的数量
  # 在初始化PPO时传入M

MAX_EP_STEPS = env.slot_num
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# ppo = PPO(s_dim, a_dim, a_bound)
ppo = PPO(s_dim, a_dim, a_bound, M)
all_ep_r = []
for ep in range(MAX_EPISODES):
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0

    for t in range(MAX_EP_STEPS):
        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a)

        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r + 8) / 8)  # normalize reward, find a suitable value

        s = s_
        ep_r += r

        if (t + 1) % BATCH_SIZE == 0 or t == MAX_EP_STEPS - 1:
            v_s_ = ppo.get_value(s_)

            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(discounted_r)
            adv = br - ppo.get_value(bs)
            ppo.update(bs, ba, br, adv)

            buffer_s, buffer_a, buffer_r = [], [], []

    if ep == 0:
        all_ep_r.append(ep_r)
    else:
        all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
    print('Episode:', ep, ' Reward: %i' % int(ep_r))

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode')
plt.ylabel('Moving averaged episode reward')
plt.show()
