import matplotlib.pyplot as plt

# 假设这是 D2PG 和 DDPG 在 80, 100, 120 Mbits 下的平均延迟数据
task_sizes = [60, 80, 100, 120]
# task_sizes = [60, 80, 100, 120, 140]
delay_d2pg = [46, 60, 76, 85]  # 示例数据，请替换为你实际的实验结果, 92
delay_ddpg = [50, 68, 79, 87]  # 示例数据，请替换为你实际的实验结果, 89
delay_edge = [65, 86, 108, 130]  # 示例数据，请替换为你实际的实验结果, 151
delay_local = [210, 280, 350, 422]  # 示例数据，请替换为你实际的实验结果, 490

# 绘制 D2PG 和 DDPG 的延迟曲线
plt.figure(figsize=(8, 6))
plt.plot(task_sizes, delay_d2pg, label='E-DDPG', linestyle='--', marker='s', color='orange')
plt.plot(task_sizes, delay_ddpg, label='DDPG', linestyle='-', marker='o', color='blue')
plt.plot(task_sizes, delay_edge, label='Edge', linestyle='-.', marker='x', color='purple')
# plt.plot(task_sizes, delay_local, label='Local', linestyle=':', marker='d', color='brown')

# 设置图表属性
plt.xlabel('Task Sizes (Mbits)')
plt.ylabel('Delay (s)')
plt.title('Average Delay Comparison: TD3 vs DDPG vs Edge')
plt.legend()
plt.grid(True)

# 保存并显示图表
plt.savefig("delay_comparison_all.png")
plt.show()

