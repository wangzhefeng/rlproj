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
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np


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




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
