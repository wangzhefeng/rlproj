# -*- coding: utf-8 -*-

# ***************************************************
# * File        : markov_reward_process.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-26
# * Version     : 1.0.082623
# * Description : description
# * Link        : https://hrl.boyuai.com/chapter/1/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E5%86%B3%E7%AD%96%E8%BF%87%E7%A8%8B
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


def compute_return(start_index, chain, rewards, gamma):
    """
    计算回报

    Args:
        start_index (_type_): _description_
        chain (_type_): _description_
        rewards (_type_): _description_
        gamma (_type_): _description_

    Returns:
        _type_: _description_
    """
    G = 0
    for i in reversed(range(start_index, len(chain))):
        G = gamma * G + rewards[chain[i] - 1]

    return G


def compute_value(P, rewards, gamma, states_num):
    """
    利用贝尔曼方程的矩阵形式计算解析解

    Args:
        P (_type_): _description_
        rewards (_type_): _description_
        gamma (_type_): _description_
        states_num (_type_): 马尔可夫决策过程的状态数
    """
    rewards = np.array(rewards).reshape((-1, 1))
    value = np.dot(
        np.linalg.inv(np.eye(states_num, states_num) - gamma * P), 
        rewards
    )

    return value




# 测试代码 main 函数
def main():
    np.random.seed(0)

    # 状态转移概率矩阵, 状态集合: S=[S1, S2, S3, S4, S5, S6]
    P = [
        [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
        [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
        [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    P = np.array(P)
    # 奖励函数, R:[R1, R2, R3, R4, R5, R6]
    rewards = [-1, -2, -2, 10, 1, 0]
    # 折扣因子
    gamma = 0.5

    # 一个状态序列: S1-S2-S3-S6
    chain = [1, 2, 3, 6]
    start_index = 0

    # 马尔可夫奖励过程(S1-S6)的回报
    G = compute_return(start_index, chain, rewards, gamma)
    logger.info(f"根据本序列计算得到回报为：{G}")
    
    # 马尔可夫奖励过程每个状态价值
    V = compute_value(P, rewards, gamma, states_num=6)
    logger.info(f"马尔可夫奖励过程每个状态价值: \n{V}")

if __name__ == "__main__":
    main()
