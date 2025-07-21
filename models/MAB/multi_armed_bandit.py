# -*- coding: utf-8 -*-

# ***************************************************
# * File        : multi_armed_bandit.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-19
# * Version     : 1.0.071901
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


class BernoulliBandit:
    """
    伯努利多臂老虎机，输入 K 表示拉杆个数
    """
    
    def __init__(self, K):
        self.probs = np.random.uniform(size=K) # 随机生成K个0~1的数，作为拉动每根拉杆的获奖概率
        self.best_idx = np.argmax(self.probs) # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx] # 最大的获奖概率
        self.K = K
    
    def step(self, k):
        """
        当玩家选择了k号拉杆后,根据拉动该老虎机的k号拉杆获得奖励的概率返回1（获奖）或0（未获奖）
        """
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
