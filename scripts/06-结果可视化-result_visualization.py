import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import glob
import platform

# --- 配置科研绘图风格 ---
def set_science_style():
    # 使用简洁的背景风格
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            plt.style.use('ggplot')

    # 字体设置：优先使用 Times New Roman (英文) + SimHei (中文)
    # 注意：Matplotlib 的字体回退机制可能需要配置
    system = platform.system()
    if system == 'Windows':
        # 增加 Microsoft YaHei (微软雅黑) 和 SimSun (宋体) 作为备选
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    else:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'sans-serif']
        
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 自定义一些参数以符合科研风格
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['figure.dpi'] = 300 # 高分辨率
    plt.rcParams['savefig.dpi'] = 300

def load_data(file_path):
    """读取潮流计算结果Excel"""
    try:
        # 读取 '节点电压(pu)' sheet
        # index_col=0 假设第一列是节点编号/名称
        df = pd.read_excel(file_path, sheet_name='节点电压(pu)', index_col=0)
        return df
    except Exception as e:
        print(f"读取文件失败 {file_path}: {e}")
        return None

def plot_voltage_bar_1d(df, date_str, hour, output_dir):
    """
    1. 24小时各时段节点电压可视化（一维柱状图）
    """
    col_name = f"{hour}点"
    if col_name not in df.columns:
        # 尝试兼容可能的列名格式
        if str(hour) in df.columns:
            col_name = str(hour)
        else:
            # print(f"  [跳过] 列 {col_name} 不存在")
            return

    data = df[col_name]
    # 过滤掉 NaN
    data = data.dropna()
    nodes = data.index

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制柱状图
    # 科研蓝: #4c72b0
    bars = ax.bar(range(len(nodes)), data.values, color='#4c72b0', alpha=0.85, edgecolor='black', linewidth=0.5, width=0.6)
    
    # 设置X轴
    ax.set_xticks(range(len(nodes)))
    
    # 智能设置X轴标签显示
    if len(nodes) > 30:
        # 如果节点太多，每隔几个显示一个，或者只显示特定节点
        step = len(nodes) // 30 + 1
        ax.set_xticks(range(0, len(nodes), step))
        ax.set_xticklabels(nodes[::step], rotation=90, fontsize=8)
    else:
        ax.set_xticklabels(nodes, rotation=45, ha='right')
        
    ax.set_xlabel('节点编号', fontweight='bold')
    ax.set_ylabel('电压标幺值 (p.u.)', fontweight='bold')
    ax.set_title(f'{date_str} - {hour}:00 节点电压分布', fontweight='bold')
    
    # 设置Y轴范围，突出电压变化 (通常在0.9-1.1之间)
    # 动态调整范围，但保持在合理区间
    min_val = data.min()
    max_val = data.max()
    y_lower = min(0.9, min_val - 0.02)
    y_upper = max(1.1, max_val + 0.02)
    ax.set_ylim(y_lower, y_upper)
    
    # 添加参考线
    ax.axhline(y=1.07, color='#d62728', linestyle='--', linewidth=1.5, label='上限 (1.07)')
    ax.axhline(y=0.93, color='#d62728', linestyle='--', linewidth=1.5, label='下限 (0.93)') 
    ax.axhline(y=1.0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    
    # 保存
    # 创建日期子文件夹以防文件过多
    date_dir = os.path.join(output_dir, date_str)
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)
        
    save_path = os.path.join(date_dir, f"1D_电压分布_{hour}点.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    # print(f"  [生成] {save_path}")

def plot_voltage_surface_3d(df, date_str, output_dir):
    """
    2. 单日24小时节点电压可视化（三维图）
    """
    # 准备数据
    hours = range(24)
    sorted_cols = []
    valid_hours = []
    
    for h in hours:
        col = f"{h}点"
        if col in df.columns:
            sorted_cols.append(col)
            valid_hours.append(h)
    
    if not sorted_cols:
        print("  [警告] 没有找到有效的时间列数据")
        return

    # 提取数值矩阵 (Nodes x Hours)
    Z = df[sorted_cols].values 
    # Z.shape is (num_nodes, num_hours)
    
    # 构建网格
    # X轴: 时间
    # Y轴: 节点索引 (使用 range 代替具体的节点名，避免轴乱)
    X_hours, Y_nodes = np.meshgrid(valid_hours, range(len(df.index)))
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制曲面
    # cmap='viridis' (蓝绿黄) 或 'coolwarm' (蓝红)
    surf = ax.plot_surface(X_hours, Y_nodes, Z, cmap='viridis', edgecolor='none', alpha=0.9, antialiased=True)
    
    ax.set_xlabel('时间 (h)', labelpad=10, fontweight='bold')
    ax.set_ylabel('节点索引', labelpad=10, fontweight='bold')
    ax.set_zlabel('电压 (p.u.)', labelpad=10, fontweight='bold')
    ax.set_title(f'{date_str} 全网电压时空分布 (3D)', fontweight='bold', pad=20)
    
    # 调整视角
    ax.view_init(elev=25, azim=-135)
    
    # 添加颜色条
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    cbar.set_label('电压标幺值 (p.u.)', fontweight='bold')
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"3D_电压时空分布_{date_str}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  [生成] {save_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    
    set_science_style()

    def process_visualization(input_folder, output_folder, file_pattern, prefix_remove, title_suffix=""):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"创建输出目录: {output_folder}")
            
        # 查找所有结果文件
        result_files = glob.glob(os.path.join(input_folder, file_pattern))
        
        if not result_files:
            print(f"在 {input_folder} 未找到符合 {file_pattern} 的结果文件。")
            return
            
        print(f"在 {input_folder} 找到 {len(result_files)} 个结果文件，开始处理...")
        
        for fpath in result_files:
            fname = os.path.basename(fpath)
            # 提取日期
            date_str = fname.replace(prefix_remove, "").replace(".xlsx", "")
            
            # 用于显示和文件夹命名的日期标识
            display_date = date_str + title_suffix
            
            print(f"正在处理: {display_date} ...")
            
            df = load_data(fpath)
            if df is not None:
                # 1. 生成3D图
                plot_voltage_surface_3d(df, display_date, output_folder)
                
                # 2. 生成1D柱状图 (生成所有24小时)
                # print(f"  正在生成24小时柱状图...")
                for h in range(24):
                     plot_voltage_bar_1d(df, display_date, h, output_folder)

    # 1. 原始潮流计算结果 (优化前无调压)
    input_folder_1 = os.path.join(base_data_dir, "优化配置结果", "优化前无调压结果", "04_潮流计算结果")
    output_folder_1 = os.path.join(base_data_dir, "优化配置结果", "优化前无调压结果", "可视化结果")
    print("--- 处理原始潮流结果 ---")
    process_visualization(input_folder_1, output_folder_1, "潮流结果_*.xlsx", "潮流结果_")

    # 2. 优化后潮流计算结果
    input_folder_2 = os.path.join(base_data_dir, "优化配置结果", "06_优化后潮流计算")
    output_folder_2 = os.path.join(input_folder_2, "可视化结果")
    print("\n--- 处理优化后潮流结果 ---")
    process_visualization(input_folder_2, output_folder_2, "优化后潮流结果_*.xlsx", "优化后潮流结果_", "(优化后)")

    print("\n所有可视化任务完成。")

if __name__ == "__main__":
    main()
