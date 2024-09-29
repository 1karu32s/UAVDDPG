import tensorflow as tf
import numpy as np
from UAV_env import UAVEnv
import time
import matplotlib.pyplot as plt
from state_normalization import StateNormalization

#####################  超参数  ####################
MAX_EPISODES = 1000
LR_A = 0.00001  # Actor 的学习率
LR_C = 0.00002  # Critic 的学习率，降低学习率
GAMMA = 0.99  # 奖励折扣因子
TAU = 0.005  # 软更新参数
VAR_MIN = 0.01
MEMORY_CAPACITY = 30000  # 增加经验回放缓冲区
BATCH_SIZE = 64
POLICY_DELAY = 4  # Actor 的延迟更新步数
POLICY_NOISE = 0.05  # 目标策略的噪声，减少噪声
NOISE_CLIP = 0.5  # 噪声剪切范围
OUTPUT_GRAPH = False
L2_REG_SCALE = 1e-4  # 正则化系数

###############################  TD3算法  ####################################
class TD3(object):
    def __init__(self, a_dim, s_dim, a_bound):
        # 初始化内存，记录损失，启动 session
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 2), dtype=np.float32)  # +2 表示存储奖励和 done 标志
        self.pointer = 0
        self.sess = tf.Session()

        # 初始化 Actor 和 Critic 网络
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')  # 当前状态
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')  # 下一状态
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')  # 奖励
        self.done = tf.placeholder(tf.float32, [None, 1], 'done')  # 结束标志

        # 存储 Critic 和 Actor 的损失
        self.critic_loss_list = []
        self.actor_loss_list = []

        # 正则化器
        regularizer = tf.contrib.layers.l2_regularizer(scale=L2_REG_SCALE)

        # Actor 网络
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)

        # Critic 网络
        with tf.variable_scope('Critic'):
            q1 = self._build_c(self.S, self.a, scope='eval1', trainable=True, regularizer=regularizer)
            q1_ = self._build_c(self.S_, a_, scope='target1', trainable=False, regularizer=regularizer)
            q2 = self._build_c(self.S, self.a, scope='eval2', trainable=True, regularizer=regularizer)
            q2_ = self._build_c(self.S_, a_, scope='target2', trainable=False, regularizer=regularizer)

        # 获取网络参数
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce1_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval1')
        self.ce2_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval2')
        self.ct1_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target1')
        self.ct2_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target2')

        # 软更新目标网络参数
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct1_params + self.ct2_params,
                                             self.ae_params + self.ce1_params + self.ce2_params)]

        # TD3的目标 Q 值计算
        min_q_ = tf.minimum(q1_, q2_)  # 使用最小的Q值来减少过高估计
        q_target = self.R + GAMMA * (1 - self.done) * min_q_
        q_target = tf.clip_by_value(q_target, -10, 10)  # 裁剪目标 Q 值

        # Critic 的损失和训练操作
        self.td_error1 = tf.losses.mean_squared_error(labels=q_target, predictions=q1)
        self.td_error2 = tf.losses.mean_squared_error(labels=q_target, predictions=q2)

        # 梯度裁剪
        optimizer1 = tf.train.AdamOptimizer(LR_C)
        grads_and_vars1 = optimizer1.compute_gradients(self.td_error1, var_list=self.ce1_params)
        clipped_grads_and_vars1 = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in grads_and_vars1]
        self.ctrain1 = optimizer1.apply_gradients(clipped_grads_and_vars1)

        optimizer2 = tf.train.AdamOptimizer(LR_C)
        grads_and_vars2 = optimizer2.compute_gradients(self.td_error2, var_list=self.ce2_params)
        clipped_grads_and_vars2 = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in grads_and_vars2]
        self.ctrain2 = optimizer2.apply_gradients(clipped_grads_and_vars2)

        # Actor 的损失（延迟更新）
        self.a_loss = - tf.reduce_mean(q1)  # Actor 的损失为负的 Q 值
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(self.a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

        if OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def choose_action(self, s):
        temp = self.sess.run(self.a, {self.S: s[np.newaxis, :]})
        return temp[0]

    def learn(self, update_policy=True):
        # 从内存中采样一批数据
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        done = bt[:, -1:]

        # 更新两个 Critic 网络并获取损失值
        _, _, critic_loss1, critic_loss2 = self.sess.run([self.ctrain1, self.ctrain2, self.td_error1, self.td_error2],
                                                         {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.done: done})

        # 将 Critic 的损失值存储
        self.critic_loss_list.append((critic_loss1 + critic_loss2) / 2)

        # 延迟更新策略网络
        if update_policy:
            _, actor_loss = self.sess.run([self.atrain, self.a_loss], {self.S: bs})
            self.actor_loss_list.append(actor_loss)
            self.sess.run(self.soft_replace)

    def store_transition(self, s, a, r, s_, done):
        # 存储状态、动作、奖励、下一状态和结束标志
        transition = np.hstack((s, a, [r], s_, [done]))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        # 构建 Actor 网络
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 400, activation=tf.nn.relu6, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, name='l3', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound[1], name='scaled_a')

    def _build_c(self, s, a, scope, trainable, regularizer=None):
        # 构建 Critic 网络
        with tf.variable_scope(scope):
            n_l1 = 400
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable, regularizer=regularizer)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable, regularizer=regularizer)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable, kernel_regularizer=regularizer)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, name='l3', trainable=trainable, kernel_regularizer=regularizer)
            return tf.layers.dense(net, 1, trainable=trainable)  # 输出 Q(s,a)

###############################  训练主循环  ####################################
np.random.seed(1)
tf.set_random_seed(1)

env = UAVEnv()
MAX_EP_STEPS = env.slot_num
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound  # [-1,1]

td3 = TD3(a_dim, s_dim, a_bound)

var = 0.01  # 控制探索噪声
t1 = time.time()
ep_reward_list = []
s_normal = StateNormalization()

for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    update_policy = False

    j = 0
    while j < MAX_EP_STEPS:
        # 加入探索噪声
        a = td3.choose_action(s_normal.state_normal(s))
        a = np.clip(np.random.normal(a, var), *a_bound)  # 加入高斯噪声用于探索
        s_, r, is_terminal, step_redo, offloading_ratio_change, reset_dist = env.step(a)
        if step_redo:
            continue
        if reset_dist:
            a[2] = -1
        if offloading_ratio_change:
            a[3] = -1

        td3.store_transition(s_normal.state_normal(s), a, r, s_normal.state_normal(s_), is_terminal)

        if td3.pointer > BATCH_SIZE:
            td3.learn(update_policy)

            # 每 POLICY_DELAY 步更新策略网络
            if j % POLICY_DELAY == 0:
                update_policy = True
            else:
                update_policy = False

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS - 1 or is_terminal:
            print(f'Episode: {i}, Steps: {j}, Reward: {ep_reward:.2f}, Explore: {var:.3f}')
            ep_reward_list.append(ep_reward)
            with open('output.txt', 'a') as file_obj:
                file_obj.write("\n======== This episode is done ========")
            break
        j += 1

# 绘制损失曲线
plt.figure()
plt.plot(td3.critic_loss_list, label='Critic Loss')
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Critic Loss Over Time")
plt.legend()
plt.show()

plt.figure()
plt.plot(td3.actor_loss_list, label='Actor Loss', color='orange')
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Actor Loss Over Time")
plt.legend()
plt.show()

# 运行时间统计
print('Running time: ', time.time() - t1)
plt.plot(ep_reward_list)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
