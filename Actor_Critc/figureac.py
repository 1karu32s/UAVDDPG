import matplotlib.pyplot as plt

# 使用绝对路径
file_path = r'D:\desktop\taskoffloading\DDPG\UAV-DDPG-main\Actor Critc\output.txt'

# 读取 output.txt 文件
with open(file_path, 'r') as file:
    lines = file.readlines()

# 解析数据
delays_ac = []
current_episode_delay = 0
for line in lines:
    line = line.strip()
    if line.startswith("delay:"):
        # 提取 delay 值并累加
        delay_value = float(line.split(":")[1])
        current_episode_delay += delay_value
    elif line.startswith("======== This episode is done ========"):
        # 当前 episode 完成，将累计延时加入 delays 列表并重置
        delays_ac.append(current_episode_delay)
        current_episode_delay = 0

# 绘制每个 episode 的累计延时图
plt.figure()
plt.plot(delays_ac)
plt.xlabel('Episode')
plt.ylabel('delay (s)')
# plt.title('每个 Episode 的累计延时')
plt.show()

