import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker

# --- 配置 ---
INPUT_FILE = 'data/优化配置结果/验证结果/对比分析_小水电节点电压.csv'
OUTPUT_DIR = 'data/优化配置结果/可视化结果/小水电节点电压对比'

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def main():
    # 1. 检查输入文件
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到输入文件 {INPUT_FILE}")
        return

    # 2. 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")

    # 3. 读取数据
    print(f"正在读取数据: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"读取CSV失败: {e}")
        return

    # 确保时刻是数字
    df['时刻'] = pd.to_numeric(df['时刻'], errors='coerce')
    df = df.sort_values(by='时刻')

    # 4. 按日期和节点分组处理
    # 获取所有日期和节点的组合
    groups = df.groupby(['日期', '节点ID'])

    for (date_str, node_id), group_df in groups:
        print(f"正在生成图表: 日期 {date_str}, 节点 {node_id}")
        
        # 准备数据 (确保按0-23小时排序)
        # 创建一个完整的0-23小时的DataFrame，防止数据缺失
        full_hours = pd.DataFrame({'时刻': range(24)})
        plot_data = pd.merge(full_hours, group_df, on='时刻', how='left')
        
        hours = plot_data['时刻']
        v_pre = plot_data['配置前电压(pu)']
        v_post = plot_data['配置后电压(pu)']
        v_improve_pct = plot_data['电压偏差改善率（仅考虑越线情况）(%)']

        # 5. 绘图
        fig, ax = plt.subplots(figsize=(12, 6))

        # --- 右轴：绘制电压偏差改善率 (柱状图) ---
        ax2 = ax.twinx()
        
        # 计算合适的Y轴范围
        if not v_improve_pct.dropna().empty:
            max_imp = v_improve_pct.max()
            min_imp = v_improve_pct.min()
            
            # 设定 Y 轴范围，使柱状图位于下方
            y2_min = min(0, min_imp)
            if y2_min < 0: y2_min = y2_min * 1.2
            
            # 上限设得高一些，让柱子比较矮
            y2_max = max(max_imp, 1.0) * 3
            
            ax2.set_ylim(y2_min, y2_max)
            
        # 绘制柱状图
        # 颜色：使用红色，代表"优化带来的变化"
        bar_color = '#d62728' # Matplotlib standard red
        bars = ax2.bar(hours, v_improve_pct, color=bar_color, alpha=0.15, label='电压偏差改善率（仅考虑越线情况）(%)', width=0.6)
        
        ax2.set_ylabel('电压偏差改善率（仅考虑越线情况） (%)', fontsize=12, color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.spines['right'].set_color('black')
        
        # --- 左轴：绘制电压曲线 ---
        # 确保左轴在右轴之上
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

        # 绘制电压标准线
        # 上限 1.07
        line_upper = ax.axhline(y=1.07, color='green', linestyle='-.', label='电压上限 (1.07)', alpha=0.8)
        # 下限 0.93
        line_lower = ax.axhline(y=0.93, color='green', linestyle=':', label='电压下限 (0.93)', alpha=0.8)

        # 绘制 优化前 (蓝色虚线 带圆点)
        line_pre, = ax.plot(hours, v_pre, color='blue', linestyle='--', marker='o', markersize=4, label='优化前电压', alpha=0.7)

        # 绘制 优化后 (红色实线 带方点 加粗)
        line_post, = ax.plot(hours, v_post, color='red', linestyle='-', marker='s', markersize=4, linewidth=2.5, label='优化后电压')

        # 6. 设置图表元素
        ax.set_title(f"小水电节点电压优化对比 - {date_str} (Node {node_id})", fontsize=14, color='black')
        ax.set_xlabel("时刻 (小时)", fontsize=12, color='black')
        ax.set_ylabel("电压 (pu)", fontsize=12, color='black')
        
        # X轴刻度 0-23
        ax.set_xticks(range(24))
        ax.set_xlim(-0.5, 23.5)

        # Y轴范围动态调整
        all_v = pd.concat([v_pre, v_post]).dropna()
        if not all_v.empty:
            y_min = min(all_v.min(), 0.9) 
            y_max = max(all_v.max(), 1.1)
            # 确保包含 1.07 标准线
            y_max = max(y_max, 1.08)
            
            y_margin = (y_max - y_min) * 0.1
            ax.set_ylim(max(0, y_min - y_margin), y_max + y_margin)
        else:
            ax.set_ylim(0.9, 1.1)

        # 网格
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 图例 (合并左右轴图例)
        lines = [line_upper, line_lower, line_pre, line_post, bars]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')

        # 7. 保存图片
        # 文件名: 电压对比_日期_节点.png
        safe_date = str(date_str).replace('/', '-').replace('\\', '-')
        filename = f"电压对比_{safe_date}_Node_{node_id}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        
        print(f"  已保存: {save_path}")

    print("\n所有图表生成完毕。")

if __name__ == "__main__":
    main()
