# -*- coding: utf-8 -*-

# ***************************************************
# * File        : solver.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-21
# * Version     : 1.0.072110
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import numpy as np

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


class Solver:
    """
    多臂老虎机算法基本框架
    """
    
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # 每根拉杆的尝试次数
        self.regret = 0.0  # 当前步的累积懊悔
        self.actions = []  # 维护一个列表，记录每一步的动作
        self.regrets = []  # 维护一个列表，记录每一步的累积懊悔

    def update_regret(self, k):
        """
        计算累积懊悔并保存，k为本次动作选择的拉杆的编号
        """
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        """
        返回当前动作选择哪一根拉杆，由每个具体的策略实现
            - 根据策略选择动作、根据动作获取奖励和更新期望奖励估值
        """
        raise NotImplementedError
    
    def run(self, num_steps):
        """
        运行一定次数，num_steps 为总运行次数
            - 更新累积懊悔和计数
        """
        for _ in range(num_steps):
            k = self.run_one_step()
            # 更新拉杆的尝试次数
            self.counts[k] += 1
            # 更新动作
            self.actions.append(k)
            # 更新累积懊悔
            self.update_regret(k)


class EpsilonGreedy(Solver):
    """
    epsilon贪婪算法(epsilon-greedy),继承 Solver类
    """
    
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        
        self.epsilon = epsilon
        # 初始化拉动所有拉杆的期望奖励估计值
        self.estimates = np.array([init_prob] * self.bandit.K)
    
    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)  # 随机选择一根拉杆
        else:
            k = np.argmax(self.estimates)  # 选择期望奖励估计值最大的拉杆
        
        r = self.bandit.step(k)  # 得到本次动作的奖励
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])

        return k



# 测试代码 main 函数
def main():
    from plots import plot_results

if __name__ == "__main__":
    main()
