# -*- coding: utf-8 -*-

# ***************************************************
# * File        : epsilon_greedy.py
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


class EpsilonGreedy(Solver):
    """
    epsilon-贪婪算法(epsilon-greedy), 继承 Solver类
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
    from models.MAB.multi_armed_bandit import BernoulliBandit

    # 设定随机种子，使实验具有可重复性
    np.random.seed(0)
    
    # 具有 10 个拉杆的多臂老虎机
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    logger.info(f"随机生成了一个 {K} 臂伯努利老虎机")
    logger.info(f"获奖概率最大的拉杆为 {bandit_10_arm.best_idx} 号, 其获奖概率为 {bandit_10_arm.best_prob:.4f}")

    # epsilon-greedy 算法: epsilon: 0.01
    epsilon_greedy_solver = EpsilonGreedy(bandit=bandit_10_arm, epsilon=0.01)
    epsilon_greedy_solver.run(num_steps=5000)
    logger.info(f"epsilon-贪婪算法的累计懊悔为：{epsilon_greedy_solver.regret}")
    plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])
    
    # epsilon-greedy 算法: 多参数实验
    epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
    epsilon_greedy_solver_list = [EpsilonGreedy(bandit=bandit_10_arm, epsilon=e) for e in epsilons]
    for solver in epsilon_greedy_solver_list:
        solver.run(5000)
    epsilon_greedy_solver_names = [f"epsilon={e}" for e in epsilons]
    plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

if __name__ == "__main__":
    main()
