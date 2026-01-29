import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免GUI窗口
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import glob
import platform
import warnings
import gc  # 垃圾回收

warnings.filterwarnings("ignore")
plt.ioff()  # 关闭交互模式

# ==================== 配置部分 ====================

# 方案配置字典
SCHEMES = {
    '方案一': {
        'name': '多点分散',
        'folder': 'data/潮流计算/方案一_多点分散_潮流结果',
        'svg_config': 'data/潮流计算/方案一_多点分散_SVG优化配置结果.csv',
        'color': '#1f77b4',  # 蓝色
        'description': '在5个节点分散配置SVG'
    },
    '方案二': {
        'name': '双核心',
        'folder': 'data/潮流计算/方案二_双核心_潮流结果',
        'svg_config': 'data/潮流计算/方案二_双核心_SVG优化配置结果.csv',
        'color': '#ff7f0e',  # 橙色
        'description': '在26、15节点配置SVG'
    },
    '方案三': {
        'name': '单核心',
        'folder': 'data/潮流计算/方案三_单核心_潮流结果',
        'svg_config': 'data/潮流计算/方案三_单核心_SVG优化配置结果.csv',
        'color': '#2ca02c',  # 绿色
        'description': '仅在26节点配置SVG'
    },
    '方案四': {
        'name': '最优组合_26_22',
        'folder': 'data/潮流计算/方案四_最优组合_26_22_潮流结果',
        'svg_config': 'data/潮流计算/方案四_最优组合_26_22_SVG优化配置结果.csv',
        'color': '#d62728',  # 红色
        'description': '优化后双节点组合'
    }
}

# 基准数据路径
BASELINE_DIR = 'data/优化配置结果/优化前无调压结果/04_潮流计算结果'

# 日期列表（6个典型日）
DATES = ['10.3', '3.24', '5.7', '7.16', '8.15', '9.27']

# 关键节点（小水电）
HYDRO_NODES = [26, 15, 44, 22, 11, 7]

# 根节点线路
ROOT_LINE_NAMES = ['Line_1']  # 根节点连接的线路名称


# ==================== 样式配置 ====================

def set_science_style():
    """配置科研绘图风格"""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            plt.style.use('ggplot')

    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    else:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'sans-serif']

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300


# ==================== 数据加载模块 ====================

def load_voltage_data(excel_path):
    """读取单个Excel的节点电压数据"""
    try:
        df = pd.read_excel(excel_path, sheet_name='节点电压(pu)', index_col=0)
        return df
    except Exception as e:
        print(f"  [错误] 读取电压数据失败 {excel_path}: {e}")
        return None


def load_comparison_data(scheme_folder, baseline_dir, date_str):
    """
    加载优化前后对比数据

    返回: {
        'optimized': {'voltage': df, 'line_p': df, 'line_q': df},
        'baseline': {'voltage': df, 'line_p': df, 'line_q': df}
    }
    """
    result = {'optimized': {}, 'baseline': {}}

    # 加载优化后数据
    opt_file = os.path.join(scheme_folder, f"优化后潮流结果_{date_str}.xlsx")
    if os.path.exists(opt_file):
        try:
            result['optimized']['voltage'] = pd.read_excel(opt_file, sheet_name='节点电压(pu)', index_col=0)
            result['optimized']['line_p'] = pd.read_excel(opt_file, sheet_name='线路有功(MW)', index_col=0)
            result['optimized']['line_q'] = pd.read_excel(opt_file, sheet_name='线路无功(MVar)', index_col=0)
        except Exception as e:
            print(f"  [警告] 读取优化后数据失败: {e}")

    # 加载基准数据
    baseline_file = os.path.join(baseline_dir, f"潮流结果_{date_str}.xlsx")
    if os.path.exists(baseline_file):
        try:
            result['baseline']['voltage'] = pd.read_excel(baseline_file, sheet_name='节点电压(pu)', index_col=0)
            result['baseline']['line_p'] = pd.read_excel(baseline_file, sheet_name='线路有功(MW)', index_col=0)
            result['baseline']['line_q'] = pd.read_excel(baseline_file, sheet_name='线路无功(MVar)', index_col=0)
        except Exception as e:
            print(f"  [警告] 读取基准数据失败: {e}")
            result['baseline'] = {'voltage': pd.DataFrame(), 'line_p': pd.DataFrame(), 'line_q': pd.DataFrame()}
    else:
        result['baseline'] = {'voltage': pd.DataFrame(), 'line_p': pd.DataFrame(), 'line_q': pd.DataFrame()}

    # 计算功率因数
    for data_type in ['optimized', 'baseline']:
        if not result[data_type].get('line_p', pd.DataFrame()).empty:
            try:
                # 转换为数值类型，非数值转为NaN
                p = result[data_type]['line_p'].apply(pd.to_numeric, errors='coerce')
                q = result[data_type]['line_q'].apply(pd.to_numeric, errors='coerce')
                s = np.sqrt(p**2 + q**2)
                result[data_type]['pf'] = np.abs(p) / s.replace(0, np.nan)
                result[data_type]['pf'].fillna(1.0, inplace=True)
            except Exception as e:
                print(f"  [警告] 计算{data_type}功率因数失败: {e}")
                result[data_type]['pf'] = pd.DataFrame()

    return result


def load_svg_config(csv_path):
    """读取SVG配置结果CSV"""
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"  [错误] 读取SVG配置失败 {csv_path}: {e}")
        return pd.DataFrame()


# ==================== 复用可视化模块（来自06脚本） ====================

def plot_voltage_bar_1d(df, date_str, hour, output_dir, scheme_name):
    """电压1D柱状图"""
    col_name = f"{hour}点"
    if col_name not in df.columns:
        if str(hour) in df.columns:
            col_name = str(hour)
        else:
            return

    data = df[col_name].dropna()
    nodes = data.index

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(range(len(nodes)), data.values, color='#4c72b0', alpha=0.85,
                   edgecolor='black', linewidth=0.5, width=0.6)

    ax.set_xticks(range(len(nodes)))

    if len(nodes) > 30:
        step = len(nodes) // 30 + 1
        ax.set_xticks(range(0, len(nodes), step))
        ax.set_xticklabels(nodes[::step], rotation=90, fontsize=8)
    else:
        ax.set_xticklabels(nodes, rotation=45, ha='right')

    ax.set_xlabel('节点编号', fontweight='bold')
    ax.set_ylabel('电压标幺值 (p.u.)', fontweight='bold')
    ax.set_title(f'{scheme_name} - {date_str} - {hour}:00 节点电压分布', fontweight='bold')

    min_val = data.min()
    max_val = data.max()
    y_lower = min(0.9, min_val - 0.02)
    y_upper = max(1.1, max_val + 0.02)
    ax.set_ylim(y_lower, y_upper)

    ax.axhline(y=1.07, color='#d62728', linestyle='--', linewidth=1.5, label='上限 (1.07)')
    ax.axhline(y=0.93, color='#d62728', linestyle='--', linewidth=1.5, label='下限 (0.93)')
    ax.axhline(y=1.0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    ax.legend(loc='upper right', frameon=True, framealpha=0.9)

    plt.tight_layout()

    date_dir = os.path.join(output_dir, date_str)
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)

    save_path = os.path.join(date_dir, f"1D_电压分布_{hour}点.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')  # 确保关闭所有图形
    gc.collect()  # 强制垃圾回收


def plot_voltage_surface_3d(df, date_str, output_dir, scheme_name):
    """电压3D时空分布图"""
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

    Z = df[sorted_cols].values
    X_hours, Y_nodes = np.meshgrid(valid_hours, range(len(df.index)))

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X_hours, Y_nodes, Z, cmap='viridis', edgecolor='none',
                           alpha=0.9, antialiased=True)

    ax.set_xlabel('时间 (h)', labelpad=10, fontweight='bold')
    ax.set_ylabel('节点索引', labelpad=10, fontweight='bold')
    ax.set_zlabel('电压 (p.u.)', labelpad=10, fontweight='bold')
    ax.set_title(f'{scheme_name} - {date_str} 全网电压时空分布 (3D)', fontweight='bold', pad=20)

    ax.view_init(elev=25, azim=-135)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    cbar.set_label('电压标幺值 (p.u.)', fontweight='bold')

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"3D_电压时空分布_{date_str}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')
    gc.collect()
    print(f"  [生成] {save_path}")


# ==================== 对比图模块（改造自12、13脚本） ====================

def plot_pf_comparison(data, date_str, line_name, output_path, scheme_name):
    """功率因数对比图"""
    # 检查数据是否存在
    if data['optimized'].get('pf') is None or data['baseline'].get('pf') is None:
        print(f"  [跳过] PF对比图 - 缺少功率因数数据")
        return

    if data['optimized']['pf'].empty or data['baseline']['pf'].empty:
        print(f"  [跳过] PF对比图 - 功率因数数据为空")
        return

    # 提取线路数据（使用第一行数据作为根节点线路）
    if len(data['optimized']['pf']) == 0 or len(data['baseline']['pf']) == 0:
        return

    pf_opt = data['optimized']['pf'].iloc[0]  # 第一条线路
    pf_base = data['baseline']['pf'].iloc[0]

    # 确保数据长度一致 - 只保留24小时
    try:
        hour_cols = [f"{h}点" for h in range(24)]
        if all(col in pf_opt.index for col in hour_cols):
            pf_opt = pf_opt[hour_cols]
            pf_base = pf_base[hour_cols]

        # 计算改善率
        pf_improve = ((pf_opt - pf_base) / pf_base * 100).fillna(0)
    except Exception as e:
        print(f"  [警告] 计算PF改善率失败: {e}")
        pf_improve = pd.Series([0] * 24, index=range(24))

    hours = range(24)

    fig, ax = plt.subplots(figsize=(12, 6))

    # 右轴：改善率柱状图
    ax2 = ax.twinx()
    max_imp = pf_improve.max()
    min_imp = pf_improve.min()
    abs_max = max(abs(max_imp), abs(min_imp), 1.0)

    y2_min = min(0, min_imp)
    if y2_min < 0: y2_min = y2_min * 1.2
    y2_max = max(max_imp, 1.0) * 3
    ax2.set_ylim(y2_min, y2_max)

    bars = ax2.bar(hours, pf_improve, color='#d62728', alpha=0.15,
                   label='提升百分比(%)', width=0.6)
    ax2.set_ylabel('提升百分比 (%)', fontsize=12, color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # 左轴：PF曲线
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    line_std = ax.axhline(y=0.95, color='green', linestyle='-.', label='合格标准 (0.95)', alpha=0.8)
    line_pre, = ax.plot(hours, pf_base, color='blue', linestyle='--', marker='o',
                        markersize=4, label='优化前 PF', alpha=0.7)
    line_post, = ax.plot(hours, pf_opt, color='red', linestyle='-', marker='s',
                         markersize=4, linewidth=2.5, label='优化后 PF')

    ax.set_title(f"{scheme_name} - 根节点功率因数对比 - {date_str}", fontsize=14)
    ax.set_xlabel("时刻 (小时)", fontsize=12)
    ax.set_ylabel("功率因数 (PF)", fontsize=12)

    ax.set_xticks(range(24))
    ax.set_xlim(-0.5, 23.5)

    all_pf = pd.concat([pf_base, pf_opt]).dropna()
    if not all_pf.empty:
        y_min = min(all_pf.min(), 0.9)
        y_max = max(all_pf.max(), 1.0)
        y_margin = (y_max - y_min) * 0.1
        ax.set_ylim(max(0, y_min - y_margin), min(1.05, y_max + y_margin))
    else:
        ax.set_ylim(0.8, 1.05)

    ax.grid(True, linestyle=':', alpha=0.6)

    lines = [line_std, line_pre, line_post, bars]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    plt.close('all')
    gc.collect()


def plot_voltage_comparison(data, date_str, node_id, output_path, scheme_name):
    """节点电压对比图"""
    # 检查数据是否存在
    if data['optimized'].get('voltage') is None or data['baseline'].get('voltage') is None:
        print(f"  [跳过] 节点{node_id}电压对比图 - 缺少电压数据")
        return

    if data['optimized']['voltage'].empty or data['baseline']['voltage'].empty:
        print(f"  [跳过] 节点{node_id}电压对比图 - 电压数据为空")
        return

    # 检查节点是否存在
    if node_id not in data['optimized']['voltage'].index or node_id not in data['baseline']['voltage'].index:
        print(f"  [跳过] 节点{node_id}不存在于数据中")
        return

    v_opt = data['optimized']['voltage'].loc[node_id]
    v_base = data['baseline']['voltage'].loc[node_id]

    # 计算改善率（电压偏差改善）
    dev_base = np.abs(v_base - 1.0)
    dev_opt = np.abs(v_opt - 1.0)

    # 确保数据类型为数值并长度一致
    try:
        # 只保留24小时的列
        hour_cols = [f"{h}点" for h in range(24)]
        if all(col in dev_base.index for col in hour_cols):
            dev_base = dev_base[hour_cols]
            dev_opt = dev_opt[hour_cols]

        v_improve = ((dev_base - dev_opt) / dev_base * 100).fillna(0)
    except Exception as e:
        print(f"  [警告] 计算改善率失败: {e}")
        v_improve = pd.Series([0] * 24, index=range(24))

    hours = range(24)

    fig, ax = plt.subplots(figsize=(12, 6))

    # 右轴：改善率柱状图
    ax2 = ax.twinx()
    max_imp = v_improve.max()
    min_imp = v_improve.min()

    y2_min = min(0, min_imp)
    if y2_min < 0: y2_min = y2_min * 1.2
    y2_max = max(max_imp, 1.0) * 3
    ax2.set_ylim(y2_min, y2_max)

    bars = ax2.bar(hours, v_improve, color='#d62728', alpha=0.15,
                   label='电压偏差改善率(%)', width=0.6)
    ax2.set_ylabel('电压偏差改善率 (%)', fontsize=12, color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # 左轴：电压曲线
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    line_upper = ax.axhline(y=1.07, color='green', linestyle='-.', label='电压上限 (1.07)', alpha=0.8)
    line_lower = ax.axhline(y=0.93, color='green', linestyle=':', label='电压下限 (0.93)', alpha=0.8)
    line_pre, = ax.plot(hours, v_base, color='blue', linestyle='--', marker='o',
                        markersize=4, label='优化前电压', alpha=0.7)
    line_post, = ax.plot(hours, v_opt, color='red', linestyle='-', marker='s',
                         markersize=4, linewidth=2.5, label='优化后电压')

    ax.set_title(f"{scheme_name} - 节点电压对比 - {date_str} (Node {node_id})", fontsize=14)
    ax.set_xlabel("时刻 (小时)", fontsize=12)
    ax.set_ylabel("电压 (pu)", fontsize=12)

    ax.set_xticks(range(24))
    ax.set_xlim(-0.5, 23.5)

    all_v = pd.concat([v_base, v_opt]).dropna()
    if not all_v.empty:
        y_min = min(all_v.min(), 0.9)
        y_max = max(all_v.max(), 1.1)
        y_max = max(y_max, 1.08)
        y_margin = (y_max - y_min) * 0.1
        ax.set_ylim(max(0, y_min - y_margin), y_max + y_margin)
    else:
        ax.set_ylim(0.9, 1.1)

    ax.grid(True, linestyle=':', alpha=0.6)

    lines = [line_upper, line_lower, line_pre, line_post, bars]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    plt.close('all')
    gc.collect()


# ==================== 新增对比分析模块 ====================

def plot_svg_capacity_comparison(schemes_data, output_path):
    """SVG容量对比柱状图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    scheme_names = []
    capacities = []
    colors = []
    node_counts = []

    for scheme_key in ['方案一', '方案二', '方案三', '方案四']:
        config = SCHEMES[scheme_key]
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                config['svg_config'])

        if os.path.exists(csv_path):
            df_svg = pd.read_csv(csv_path)
            total_capacity = df_svg['建议选型_Mvar'].sum()
            node_count = len(df_svg[df_svg['建议选型_Mvar'] > 0])
        else:
            total_capacity = 0
            node_count = 0

        scheme_names.append(config['name'])
        capacities.append(total_capacity)
        colors.append(config['color'])
        node_counts.append(node_count)

    bars = ax.bar(scheme_names, capacities, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    for bar, capacity, node_count in zip(bars, capacities, node_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{capacity:.2f} MVar\n({node_count}个节点)',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_title('四种方案SVG总容量对比', fontsize=14, fontweight='bold')
    ax.set_xlabel('方案名称', fontsize=12, fontweight='bold')
    ax.set_ylabel('SVG总容量 (MVar)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    plt.close('all')
    gc.collect()
    print(f"  [生成] {output_path}")


def calculate_voltage_quality_metrics(voltage_df):
    """计算电压质量指标"""
    # 只保留24小时的列 (0点到23点)
    hour_cols = [f"{h}点" for h in range(24)]
    valid_cols = [col for col in hour_cols if col in voltage_df.columns]

    if not valid_cols:
        # 尝试数字列名
        valid_cols = [col for col in voltage_df.columns if str(col) in [str(h) for h in range(24)]]

    if not valid_cols:
        print(f"  [警告] 未找到有效的时间列，使用全部数值列")
        # 使用所有数值列
        voltage_data = voltage_df.select_dtypes(include=[np.number])
    else:
        voltage_data = voltage_df[valid_cols]

    # 转换为数值类型
    voltage_data = voltage_data.apply(pd.to_numeric, errors='coerce')

    violations = ((voltage_data > 1.07) | (voltage_data < 0.93)).sum().sum()
    avg_deviation = ((voltage_data - 1.0).abs().mean().mean()) * 100
    qualified = ((voltage_data >= 0.93) & (voltage_data <= 1.07)).sum().sum()
    total = voltage_data.size
    qualified_rate = (qualified / total) * 100 if total > 0 else 0

    return {
        'violations': int(violations),
        'avg_dev': float(avg_deviation),
        'qualified_rate': float(qualified_rate)
    }


def plot_voltage_quality_comparison(schemes_data, output_dir):
    """电压质量对比图（三合一）"""
    # 图1: 电压越限时段数对比
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(DATES))
    width = 0.2

    for i, scheme_key in enumerate(['方案一', '方案二', '方案三', '方案四']):
        if scheme_key not in schemes_data:
            continue

        violations = []
        for date in DATES:
            if date in schemes_data[scheme_key]:
                violations.append(schemes_data[scheme_key][date]['metrics']['violations'])
            else:
                violations.append(0)

        ax1.bar(x + i*width, violations, width,
               label=SCHEMES[scheme_key]['name'],
               color=SCHEMES[scheme_key]['color'],
               alpha=0.8)

    ax1.set_xlabel('日期', fontsize=12, fontweight='bold')
    ax1.set_ylabel('越限时段数 (节点×小时)', fontsize=12, fontweight='bold')
    ax1.set_title('各方案电压越限时段数对比', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(DATES)
    ax1.legend()
    ax1.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    save_path1 = os.path.join(output_dir, "电压质量对比_越限时段数.png")
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    plt.close()
    plt.close('all')
    gc.collect()
    print(f"  [生成] {save_path1}")

    # 图2: 平均电压偏差对比
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    for scheme_key in ['方案一', '方案二', '方案三', '方案四']:
        if scheme_key not in schemes_data:
            continue

        deviations = []
        for date in DATES:
            if date in schemes_data[scheme_key]:
                deviations.append(schemes_data[scheme_key][date]['metrics']['avg_dev'])
            else:
                deviations.append(0)

        ax2.plot(DATES, deviations,
                marker='o',
                label=SCHEMES[scheme_key]['name'],
                color=SCHEMES[scheme_key]['color'],
                linewidth=2.5,
                markersize=8)

    ax2.set_xlabel('日期', fontsize=12, fontweight='bold')
    ax2.set_ylabel('平均电压偏差 (%)', fontsize=12, fontweight='bold')
    ax2.set_title('各方案平均电压偏差对比', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    save_path2 = os.path.join(output_dir, "电压质量对比_平均偏差.png")
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    plt.close()
    plt.close('all')
    gc.collect()
    print(f"  [生成] {save_path2}")


# ==================== 主流程控制 ====================

def process_single_scheme(scheme_key, scheme_config, baseline_dir, base_output_dir):
    """处理单个方案的完整可视化"""
    scheme_name = scheme_config['name']
    scheme_folder = scheme_config['folder']

    print(f"\n{'='*60}")
    print(f"处理 {scheme_key} - {scheme_name}")
    print(f"{'='*60}")

    # 完整路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    scheme_folder_full = os.path.join(project_dir, scheme_folder)
    baseline_dir_full = os.path.join(project_dir, baseline_dir)

    # 创建输出目录
    scheme_output = os.path.join(base_output_dir, f"{scheme_key}_{scheme_name}")
    dirs = {
        '1d': os.path.join(scheme_output, '01_电压分布', '1D柱状图'),
        '3d': os.path.join(scheme_output, '01_电压分布', '3D时空图'),
        'pf': os.path.join(scheme_output, '02_功率因数对比'),
        'voltage_comp': os.path.join(scheme_output, '03_节点电压对比')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    scheme_data = {}

    # 遍历6个日期
    for date_str in DATES:
        print(f"\n  处理日期: {date_str}")

        # 加载优化后数据
        opt_file = os.path.join(scheme_folder_full, f"优化后潮流结果_{date_str}.xlsx")
        if not os.path.exists(opt_file):
            print(f"  [跳过] 文件不存在: {opt_file}")
            continue

        voltage_df = load_voltage_data(opt_file)
        if voltage_df is None:
            continue

        # 1. 电压1D柱状图（24小时）
        print(f"    生成24小时1D柱状图...")
        for hour in range(24):
            plot_voltage_bar_1d(voltage_df, date_str, hour, dirs['1d'], scheme_name)
            if hour % 6 == 0:  # 每6小时清理一次内存
                gc.collect()

        # 2. 电压3D时空图
        print(f"    生成3D时空分布图...")
        plot_voltage_surface_3d(voltage_df, date_str, dirs['3d'], scheme_name)

        # 3. 加载对比数据
        data = load_comparison_data(scheme_folder_full, baseline_dir_full, date_str)

        # 4. 功率因数对比图
        if not data['baseline']['voltage'].empty:
            print(f"    生成功率因数对比图...")
            for line_name in ROOT_LINE_NAMES:
                output_path = os.path.join(dirs['pf'], f"PF对比_{date_str}_{line_name}.png")
                plot_pf_comparison(data, date_str, line_name, output_path, scheme_name)

        # 5. 节点电压对比图
        if not data['baseline']['voltage'].empty:
            print(f"    生成节点电压对比图...")
            for node_id in HYDRO_NODES:
                output_path = os.path.join(dirs['voltage_comp'], f"电压对比_{date_str}_Node_{node_id}.png")
                plot_voltage_comparison(data, date_str, node_id, output_path, scheme_name)

        # 保存数据用于跨方案对比
        metrics = calculate_voltage_quality_metrics(voltage_df)
        scheme_data[date_str] = {
            'voltage': voltage_df,
            'metrics': metrics
        }

        # 清理内存
        gc.collect()
        print(f"  日期 {date_str} 处理完成\n")

    print(f"\n{scheme_key} 处理完成")
    return scheme_data


def generate_cross_scheme_comparison(all_schemes_data, output_dir):
    """生成跨方案对比图表"""
    print(f"\n{'='*60}")
    print("生成跨方案对比图表")
    print(f"{'='*60}")

    comparison_dir = os.path.join(output_dir, "00_跨方案对比")
    os.makedirs(comparison_dir, exist_ok=True)

    # 1. SVG容量对比图
    print("\n  生成SVG容量对比图...")
    svg_capacity_path = os.path.join(comparison_dir, "SVG容量对比.png")
    plot_svg_capacity_comparison(all_schemes_data, svg_capacity_path)

    # 2. 电压质量对比图
    print("\n  生成电压质量对比图...")
    plot_voltage_quality_comparison(all_schemes_data, comparison_dir)

    print(f"\n跨方案对比图表生成完成")


def main():
    """主函数：依次处理4个方案，最后生成对比图"""
    print("="*60)
    print("多方案SVG优化可视化程序")
    print("="*60)

    # 初始化
    set_science_style()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    base_output_dir = os.path.join(project_dir, 'data', '潮流计算', '多方案对比可视化结果')
    baseline_dir = BASELINE_DIR

    # 收集所有方案数据
    all_schemes_data = {}

    # 处理各方案
    for scheme_key in ['方案一', '方案二', '方案三', '方案四']:
        print(f"\n{'='*60}")
        print(f"开始处理: {scheme_key} - {SCHEMES[scheme_key]['name']}")
        print(f"{'='*60}")
        try:
            scheme_data = process_single_scheme(
                scheme_key,
                SCHEMES[scheme_key],
                baseline_dir,
                base_output_dir
            )
            all_schemes_data[scheme_key] = scheme_data
            gc.collect()  # 每个方案处理完后清理内存
        except Exception as e:
            print(f"\n[错误] 处理{scheme_key}时出错: {e}")
            import traceback
            traceback.print_exc()

    # 生成跨方案对比
    try:
        generate_cross_scheme_comparison(all_schemes_data, base_output_dir)
    except Exception as e:
        print(f"\n[错误] 生成跨方案对比时出错: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print("所有可视化任务完成！")
    print(f"{'='*60}")
    print(f"结果保存在: {base_output_dir}")

    # 统计输出
    print("\n输出统计:")
    for scheme_key, scheme_data in all_schemes_data.items():
        print(f"  {scheme_key}: 处理了 {len(scheme_data)} 个日期的数据")


if __name__ == "__main__":
    main()
