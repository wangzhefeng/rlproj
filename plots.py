# -*- coding: utf-8 -*-

# ***************************************************
# * File        : plot_process.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-21
# * Version     : 1.0.072117
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
from typing import List
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei', 'Arial Unicode MS'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.autolayout'] = True # layout
plt.rcParams['axes.grid'] = True # grid
plt.rc(
    "figure",
    autolayout=True,
    figsize=(8.0, 4.5),
    titleweight="bold",
    titlesize=18,
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
line_plot_params = dict(
    color="C0",
    linestyle="-",
    linewidth=2,
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    # legend=True,
    # label="",
)
# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def plot_results(solvers: List, solver_names: List[str]):
    """
    生成累积懊悔随时间变化的图像
    
    Args:
        solvers (List): 列表中的每个元素是一种特定的策略
        solver_names (List(str)): 存储每个策略的名称
    """
    for solver, solver_name in zip(solvers, solver_names):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_name, 
            # **line_plot_params
        )
    plt.xlabel("Time steps")
    plt.ylabel("Cumlative regrets")
    plt.title(f"{solvers[0].bandit.K}-armed bandit")
    plt.legend()
    # plt.grid(visible=True)
    plt.show()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
