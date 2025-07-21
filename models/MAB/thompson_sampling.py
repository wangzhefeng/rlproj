# -*- coding: utf-8 -*-

# ***************************************************
# * File        : thompson_sampling.py
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


class ThompsonSampling(Solver):
    """
    汤普森采样算法, 继承 Solver 类
    """

    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)

        self._a = np.ones(self.bandit.K)  # 列表, 表示每根拉杆奖励为 1 的次数
        self._b = np.ones(self.bandit.K)  # 列表, 表示每根拉杆奖励为 0 的次数
    
    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)  # 按照 Beta 分布采样一组奖励样本
        k = np.argmax(samples)
        r = self.bandit.step(k)

        self._a[k] += r  # 更新 Beta 分布的第一个参数
        self._b[k] += (1 - r)  # 更新 Beta 分布的第二个参数

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
    thompson_sampling_solver = ThompsonSampling(bandit=bandit_10_arm)
    thompson_sampling_solver.run(num_steps=5000)
    logger.info(f"Thompson sampling 算法的累计懊悔为：{thompson_sampling_solver.regret}")
    plot_results([thompson_sampling_solver], ["ThompsonSampling"])

if __name__ == "__main__":
    main()
