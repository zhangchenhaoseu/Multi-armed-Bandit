# 靡不有初，鲜克有终
# 开发时间：2022/4/28 20:49
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

step = 1000
runs = 2000
sigma = 1  # 10-armed testbed 的正态分布标准差
c = 0.5
miu_lst = [0.2, -0.85, 1.55, 0.3, 1.2, -1.5, -0.2, -1.0, 0.9, -0.6]  # 10-armed testbed 的正态分布均值列表


def action_reward(action):  # 10-armed 赌博机testbed，输入动作序号，输出对应的奖励值
    reward = miu_lst[action]+np.random.normal(0, scale=sigma)
    return reward


def random_index(rate):  # 输入概率列表，按照列表的概率输出对应动作的索引
    # """随机变量的概率函数"""
    # 参数rate为list<int>
    # 返回概率动作的下标索引
    start = 0
    index = 0
    randnum = random.randint(1, sum(rate))
    for index, scope in enumerate(rate):
        start += scope
        if randnum <= start:
            break
    return index


def random_index_UCB(Q_lst, N_list):  # 输入收益列表和动作选择次数列表，按照UCB规则输出对应动作的索引
    # """随机变量的概率函数"""
    # 参数rate为list<int>
    # 返回概率动作的下标索引
    zero_lst = []
    if 0 in N_list:
        for i in range(0, len(N_list)):
            if N_list[i] == 0:
                zero_lst.append(i)
        index = random.sample(zero_lst, 1)[0]
    else:
        index_lst = []
        max_n = max(Q_lst)
        for i in range(len(Q_lst)):
            if Q_lst[i] == max_n:
                index_lst.append(i)
        index = random.sample(index_lst, 1)[0]
    return index


def max_index(lst):  # 寻找列表中最大值的索引，如果有多个，则随机输出一个索引
    index_lst = []
    max_n = max(lst)
    for i in range(len(lst)):
        if lst[i] == max_n:
            index_lst.append(i)
    max_index = random.sample(index_lst, 1)[0]
    return max_index


def greedy(epsilon, step, runs):  # 贪婪策略，输入epsilon、步长step、平均奖励列表Q、计数列表N、概率列表P,总独立测试次数，输出平均奖励
    Q_runs_lst = [[] for m in range(runs)]  # 二维列表，存放所有runs的所有step的平均奖励Q
    Q_runs_lst_average = [0 for n in range(step)]  # 一维列表，将 Q_runs_lst 中的所有runs的对应step的Q求平均
    for run_times in range(0, runs):
        Q = [0 for i in range(10)]  # 奖励均值(是用来做greed决策的，而不是最终图里面展示的average reward)
        N = [0 for j in range(10)]  # 计数列表
        P = [0 for k in range(10)]  # 概率列表
        k = 1
        while k <= step:
            for i in range(0, 10):
                P[i] = 1000*epsilon/10
            max_ = max_index(Q)
            P[max_] = 1000 - 1000 * epsilon + P[max_]
            action_id = random_index(P)  # 得到在概率列表P下的动作序号
            N[action_id] = N[action_id] + 1
            act_reward = action_reward(action_id)  # 这一次奖励的数值（用于图像展示）
            Q_runs_lst[run_times].append(act_reward)
            Q[action_id] = Q[action_id] + (1/N[action_id])*(act_reward - Q[action_id])

            k += 1
    # print(Q_runs_lst)
    for i in range(0, step):
        for j in range(0, runs):
            Q_runs_lst_average[i] = Q_runs_lst_average[i] + Q_runs_lst[j][i]
        Q_runs_lst_average[i] = Q_runs_lst_average[i]/runs
    # print(Q_runs_lst_average)
    return Q_runs_lst_average


def UCB(epsilon, step, runs):  # 贪婪策略，输入epsilon、步长step、平均奖励列表Q、计数列表N、概率列表P,总独立测试次数，输出平均奖励
    Q_runs_lst = [[] for m in range(runs)]  # 二维列表，存放所有runs的所有step的平均奖励Q
    Q_runs_lst_average = [0 for n in range(step)]  # 一维列表，将 Q_runs_lst 中的所有runs的对应step的Q求平均
    for run_times in range(0, runs):
        Q = [0 for i in range(10)]  # 奖励均值(是用来做greed决策的，而不是最终图里面展示的average reward)
        N = [0 for j in range(10)]  # 计数列表
        P = [0 for k in range(10)]  # 概率列表
        Q_UCB = [0 for l in range(10)]  # UBC action section的奖励列表
        k = 1
        while k <= step:
            action_id = random_index_UCB(Q_UCB, N)  # 得到在概率列表P下的动作序号
            N[action_id] = N[action_id] + 1
            act_reward = action_reward(action_id)  # 这一次奖励的数值（用于图像展示）
            Q_runs_lst[run_times].append(act_reward)
            Q[action_id] = Q[action_id] + (1/N[action_id])*(act_reward - Q[action_id])
            Q_UCB[action_id] = Q[action_id] + c*(math.log(k)/N[action_id])**0.5
            k += 1
    # print(Q_runs_lst)
    for i in range(0, step):
        for j in range(0, runs):
            Q_runs_lst_average[i] = Q_runs_lst_average[i] + Q_runs_lst[j][i]
        Q_runs_lst_average[i] = Q_runs_lst_average[i]/runs
    # print(Q_runs_lst_average)
    return Q_runs_lst_average


_UCB_greedy = UCB(0, step, runs)
_010_greedy = greedy(0.1, step, runs)


x = np.arange(step)
plt.plot(x, _UCB_greedy, color='red')
plt.plot(x, _010_greedy, color='blue')
plt.legend(["UCB c=0.5", "0.1-greedy"])
plt.grid(color='gray', linestyle=':')
plt.xlabel("Step", fontweight='bold', fontsize=14, labelpad=2)
plt.ylabel('Average reward', fontweight='bold', fontsize=14, labelpad=2)
plt.show()

