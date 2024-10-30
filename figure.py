import matplotlib.pyplot as plt
import numpy as np

# 读取数据的函数
def read_rewards(filename):
    with open(filename, 'r') as file:
        return [float(line.strip()) for line in file.readlines()]

# 读取各实验的 reward 数据
reward_ac = read_rewards('D:\\desktop\\taskoffloading\\DDPG\\UAV-DDPG-main\\Actor_Critc\\reward_ac.txt')
reward_d2pg = read_rewards('D:\\desktop\\taskoffloading\\DDPG\\UAV-DDPG-main\\D2PG\\reward_d2pg.txt')
reward_ddpg_new = read_rewards('D:\\desktop\\taskoffloading\\DDPG\\UAV-DDPG-main\\D2PG\\reward_ddpg.txt')
reward_dqn = read_rewards('D:\\desktop\\taskoffloading\\DDPG\\UAV-DDPG-main\\DQN\\reward_dqn.txt')

# 对所有实验的数据进行统一长度
min_length = min(len(reward_ac), len(reward_d2pg), len(reward_ddpg_new), len(reward_dqn), 1000)  # 限制长度到1000
reward_ac = reward_ac[:min_length]
reward_d2pg = reward_d2pg[:min_length]
reward_ddpg_new = reward_ddpg_new[:min_length]
reward_dqn = reward_dqn[:min_length]

# 数据平滑处理函数
def smooth(data, window_size=10):
    box = np.ones(window_size) / window_size
    return np.convolve(data, box, mode='same')

# 设置基线奖励值
edge_only_reward = -108.7574
local_only_reward = -349.5253

# 绘制对比图
plt.figure(figsize=(10, 6))

plt.plot(smooth(reward_d2pg), label='E-DDPG', linestyle='-', marker='s', markersize=4, alpha=0.7, markevery=20)
plt.plot(smooth(reward_ddpg_new), label='DDPG', linestyle='-', marker='^', markersize=4, alpha=0.7, markevery=20)
plt.plot(smooth(reward_dqn), label='DQN', linestyle='-', marker='x', markersize=4, alpha=0.7, markevery=20)
plt.plot(smooth(reward_ac), label='AC', linestyle='-', marker='o', markersize=4, alpha=0.7, markevery=20)

# 添加水平直线
plt.axhline(y=edge_only_reward, color='purple', linestyle='--', linewidth=2, label='Edge Only')
plt.axhline(y=local_only_reward, color='brown', linestyle='--', linewidth=2, label='Local Only')

# 设置x轴限制
plt.xlim([0, 1000])

# 设置图表
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Comparison of Rewards Across Different Experiments')
plt.legend()
plt.grid(True)
plt.savefig("reward_comparison.png")  # 保存为图像
plt.show()
