import numpy as np
import tensorflow as tf
from UAV_env import UAVEnv
import time
from state_normalization import StateNormalization
import matplotlib.pyplot as plt

MAX_EPISODES = 1000
MEMORY_CAPACITY = 20000  # 增大经验回放池
BATCH_SIZE = 64


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.001,  # 调低学习率
            reward_decay=0.95,    # 调高奖励折扣因子
            e_greedy=0.9,         # 探索率最大值调整为0.9
            replace_target_iter=125,  # 增加目标网络的更新频率
            memory_size=MEMORY_CAPACITY,
            batch_size=BATCH_SIZE,
            e_greedy_increment=0.0001,  # 启用epsilon递增策略
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + 2), dtype=np.float32)  # memory里存放当前和下一个state，动作和奖励

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 128, tf.nn.relu6, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            e2 = tf.layers.dense(e1, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e2')
            self.q_eval = tf.layers.dense(e2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 128, tf.nn.relu6, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            t2 = tf.layers.dense(t1, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t2')
            self.q_next = tf.layers.dense(t2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t3')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, a, [r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = min(self.epsilon + self.epsilon_increment, self.epsilon_max)
        self.learn_step_counter += 1


if __name__ == '__main__':
    env = UAVEnv()
    Normal = StateNormalization()  # 输入状态归一化
    np.random.seed(1)
    tf.set_random_seed(1)
    s_dim = env.state_dim
    n_actions = env.n_actions
    DQN = DeepQNetwork(n_actions, s_dim, output_graph=False)
    t1 = time.time()
    ep_reward_list = []
    MAX_EP_STEPS = env.slot_num
    for i in range(MAX_EPISODES):
        # initial observation
        s = env.reset()
        ep_reward = 0
        j = 0
        while j < MAX_EP_STEPS:

            # RL
            # RL choose action based on observation
            a = DQN.choose_action(Normal.state_normal(s))
            # RL take action and get next observation and reward
            s_, r, is_terminal, step_redo, reset_offload_ratio = env.step(a)
            if step_redo:
                continue
            if reset_offload_ratio:
                # 卸载比率重新设置为0
                t1 = a % 11
                a = a - t1
            DQN.store_transition(Normal.state_normal(s), a, r, Normal.state_normal(s_))

            if DQN.memory_counter > MEMORY_CAPACITY:
                DQN.learn()

            # swap observation
            s = s_
            ep_reward += r

            if j == MAX_EP_STEPS - 1 or is_terminal:
                print('Episode:', i, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward, 'Explore: %.3f' % DQN.epsilon)
                ep_reward_list.append(ep_reward)
                with open('output.txt', 'a') as file_obj:
                    file_obj.write("\n======== This episode is done ========")  # 本episode结束
                break

            j = j + 1
print('Running time: ', time.time() - t1)
plt.plot(ep_reward_list)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("dqn.png")
plt.show()
with open("reward_dqn.txt", "w") as file:
    for reward in ep_reward_list:
        file.write(f"{reward}\n")
