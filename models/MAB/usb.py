# -*- coding: utf-8 -*-

# ***************************************************
# * File        : usb.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-21
# * Version     : 1.0.072122
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

from models.MAB.solver import Solver

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


class UCB(Solver):
    """
    UCB 算法, 继承 Solver 类
    """

    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)

        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        # 计算上置信界
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1)))
        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])

        return k




# 测试代码 main 函数
def main():
    from plots import plot_results
    from models.MAB.multi_armed_bandit import BernoulliBandit

    # 设定随机种子，使实验具有可重复性
    np.random.seed(1)
    
    # 具有 10 个拉杆的多臂老虎机
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    logger.info(f"随机生成了一个 {K} 臂伯努利老虎机")
    logger.info(f"获奖概率最大的拉杆为 {bandit_10_arm.best_idx} 号, 其获奖概率为 {bandit_10_arm.best_prob:.4f}")

    # epsilon-greedy 算法: epsilon: 0.01
    coef = 1  # 控制不确定性比重的系数
    UCB_solver = UCB(bandit=bandit_10_arm, coef=coef)
    UCB_solver.run(num_steps=5000)
    logger.info(f"UCB 算法的累计懊悔为：{UCB_solver.regret}")
    plot_results([UCB_solver], ["UCB"])

if __name__ == "__main__":
    main()
