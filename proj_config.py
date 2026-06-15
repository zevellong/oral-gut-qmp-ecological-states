# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 14:42:43 2025

@author: Zhengwu Long <longzhengwu2236@gmail.com>
"""

# config.py
from pathlib import Path
import os
import matplotlib.pyplot as plt

SYSTEM = os.name
# 这里定义了不同操作系统的项目基础路径，确保代码在不同环境下都能正确找到资源。如果需要运行，你应该先修改这里。
BASE_PATHS = {
    'nt': Path(r"C:\Users\longz\OneDrive\HUST_Working\ChenWorking\proj_oral_gut"),
    'posix': Path(r"/home/zw/zwLearn/zwProjHust/proj_oral_gut")
}

proj_base_path = BASE_PATHS.get(SYSTEM, BASE_PATHS['posix'])  # 默认使用Linux路径


# def set_pltrc(obj):
#     obj.rcParams.update({
#         'font.family': 'sans-serif',
#         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
#         'font.size': 7,          # 全局基础字体大小 (7pt 极其标准)
#         'axes.labelsize': 8,     # 坐标轴标签大小
#         'axes.titlesize': 8,     # 子图标题大小
#         'xtick.labelsize': 7,    # X轴刻度大小
#         'ytick.labelsize': 7,    # Y轴刻度大小
#         'legend.fontsize': 7,    # 图例字体大小
#         'legend.title_fontsize': 8,
#         'pdf.fonttype': 42,      # 强制 PDF 导出真实的字体文本 (极度重要!)
#         'ps.fonttype': 42,
#         'axes.linewidth': 0.8,   # 坐标轴线宽变细，显得更高级
#         'patch.linewidth': 0.8,  # 图形边框线宽
#         'lines.linewidth': 1.2   # 折线图线宽
#     })


    
def set_journal_style():
    # 静音字体报错
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'DejaVu Sans', 'Arial'], # Helvetica 与你的 LaTeX 对齐 
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.5,
        'pdf.fonttype': 42
    })

# 统一 A/B/C 标签样式
PANEL_LABEL_STYLE = dict(
    fontsize=20, 
    fontweight='bold', 
    va='bottom', 
    ha='right'
)


