import pandas as pd
import numpy as np
import os
import glob

# ==================== 配置部分 ====================

# 方案配置字典
SCHEMES = {
    '方案一': {
        'name': '多点分散',
        'folder': 'data/潮流计算/方案一_多点分散_潮流结果',
        'description': '在5个节点分散配置SVG'
    },
    '方案二': {
        'name': '双核心',
        'folder': 'data/潮流计算/方案二_双核心_潮流结果',
        'description': '在26、15节点配置SVG'
    },
    '方案三': {
        'name': '单核心',
        'folder': 'data/潮流计算/方案三_单核心_潮流结果',
        'description': '仅在26节点配置SVG'
    },
    '方案四': {
        'name': '最优组合',
        'folder': None,  # 将动态查找
        'description': '优化后双节点组合'
    }
}

def calculate_pf(p, q):
    """计算功率因数"""
    s = np.sqrt(p**2 + q**2)
    # 避免除以零
    pf = np.abs(p) / s
    pf = pf.fillna(1.0) # 如果S为0，PF设为1
    return pf

def find_scheme4_folder():
    """动态查找方案四的文件夹（格式：方案四_最优组合_X_Y_潮流结果）"""
    base_dir = 'data/潮流计算'
    pattern = os.path.join(base_dir, '方案四_最优组合_*_*_潮流结果')
    matches = glob.glob(pattern)

    if matches:
        # 返回相对路径
        return matches[0].replace('\\', '/')
    else:
        print("  [警告] 未找到方案四的潮流结果文件夹")
        return None

def process_scheme(scheme_key, scheme_config, res_pre_no_reg_dir, res_pre_with_reg_dir,
                   topology_file, root_node_id, hydro_nodes, dates, output_dir):
    """处理单个方案的对比分析

    参数:
        scheme_key: str, 方案键名（如'方案一'）
        scheme_config: dict, 方案配置
        res_pre_no_reg_dir: str, 优化前无调压结果目录
        res_pre_with_reg_dir: str, 优化前有调压结果目录
        topology_file: str, 拓扑文件路径
        root_node_id: int, 根节点ID
        hydro_nodes: list, 小水电节点列表
        dates: list, 日期列表
        output_dir: str, 输出目录

    返回:
        dict: {'task1': df_task1, 'task2': df_task2}
    """
    scheme_name = scheme_config['name']
    res_post_dir = scheme_config['folder']

    print(f"\n{'='*60}")
    print(f"处理 {scheme_key} - {scheme_name}")
    print(f"配置说明: {scheme_config['description']}")
    print(f"{'='*60}")

    if res_post_dir is None:
        print(f"  [跳过] {scheme_key} 的潮流结果文件夹未配置")
        return {'task1': None, 'task2': None}

    if not os.path.exists(res_post_dir):
        print(f"  [跳过] 找不到潮流结果文件夹: {res_post_dir}")
        return {'task1': None, 'task2': None}

    # 读取拓扑信息（只需读取一次）
    try:
        df_branch = pd.read_csv(topology_file)
    except UnicodeDecodeError:
        df_branch = pd.read_csv(topology_file, encoding='gbk')

    # 找到连接根节点的线路
    root_lines_df = df_branch[(df_branch['from_bus'] == root_node_id) | (df_branch['to_bus'] == root_node_id)]
    root_line_names = [f"Line {int(bid)}" for bid in root_lines_df['branch_id']]
    print(f"找到连接根节点(Node {root_node_id})的线路: {root_line_names}")

    # 结果容器
    task1_results = []  # 根节点线路PF
    task2_results = []  # 小水电节点电压

    for date_str in dates:
        print(f"  处理日期: {date_str}")

        file_pre_no_reg = os.path.join(res_pre_no_reg_dir, f"潮流结果_{date_str}.xlsx")
        file_pre_with_reg = os.path.join(res_pre_with_reg_dir, f"潮流结果_{date_str}.xlsx")
        file_post = os.path.join(res_post_dir, f"优化后潮流结果_{date_str}.xlsx")

        if not os.path.exists(file_pre_no_reg):
            print(f"    警告: 找不到配置前(无调压)结果文件")
            continue
        if not os.path.exists(file_post):
            print(f"    警告: 找不到配置后结果文件")
            continue

        # --- 任务一：根节点线路功率因数统计 ---
        try:
            # 读取线路有功/无功
            df_line_p_pre_no = pd.read_excel(file_pre_no_reg, sheet_name='线路有功(MW)', index_col='线路名称')
            df_line_q_pre_no = pd.read_excel(file_pre_no_reg, sheet_name='线路无功(MVar)', index_col='线路名称')

            # 配置前(有调压) - 可选
            if os.path.exists(file_pre_with_reg):
                df_line_p_pre_with = pd.read_excel(file_pre_with_reg, sheet_name='线路有功(MW)', index_col='线路名称')
                df_line_q_pre_with = pd.read_excel(file_pre_with_reg, sheet_name='线路无功(MVar)', index_col='线路名称')
            else:
                df_line_p_pre_with = None
                df_line_q_pre_with = None

            # 配置后
            df_line_p_post = pd.read_excel(file_post, sheet_name='线路有功(MW)', index_col='线路名称')
            df_line_q_post = pd.read_excel(file_post, sheet_name='线路无功(MVar)', index_col='线路名称')

            # 去除重复索引
            df_line_p_pre_no = df_line_p_pre_no[~df_line_p_pre_no.index.duplicated(keep='first')]
            df_line_q_pre_no = df_line_q_pre_no[~df_line_q_pre_no.index.duplicated(keep='first')]

            if df_line_p_pre_with is not None:
                df_line_p_pre_with = df_line_p_pre_with[~df_line_p_pre_with.index.duplicated(keep='first')]
                df_line_q_pre_with = df_line_q_pre_with[~df_line_q_pre_with.index.duplicated(keep='first')]

            df_line_p_post = df_line_p_post[~df_line_p_post.index.duplicated(keep='first')]
            df_line_q_post = df_line_q_post[~df_line_q_post.index.duplicated(keep='first')]

            # 筛选根节点线路
            for line_name in root_line_names:
                if line_name in df_line_p_pre_no.index and line_name in df_line_p_post.index:
                    # 提取24小时数据
                    p_pre_no = df_line_p_pre_no.loc[line_name]
                    q_pre_no = df_line_q_pre_no.loc[line_name]

                    if df_line_p_pre_with is not None and line_name in df_line_p_pre_with.index:
                        p_pre_with = df_line_p_pre_with.loc[line_name]
                        q_pre_with = df_line_q_pre_with.loc[line_name]
                    else:
                        p_pre_with = pd.Series([np.nan]*24, index=p_pre_no.index)
                        q_pre_with = pd.Series([np.nan]*24, index=q_pre_no.index)

                    p_post = df_line_p_post.loc[line_name]
                    q_post = df_line_q_post.loc[line_name]

                    # 计算PF
                    pf_pre_no = calculate_pf(p_pre_no, q_pre_no)
                    pf_pre_with = calculate_pf(p_pre_with, q_pre_with)
                    pf_post = calculate_pf(p_post, q_post)

                    # 存入结果
                    for hour in range(24):
                        col_name = f"{hour}点"
                        if col_name in pf_pre_no.index:
                            task1_results.append({
                                '方案': scheme_key,
                                '日期': date_str,
                                '线路名称': line_name,
                                '时刻': hour,
                                '配置前(无调压)PF': pf_pre_no[col_name],
                                '配置前(无调压)有功(MW)': p_pre_no[col_name],
                                '配置前(无调压)无功(MVar)': q_pre_no[col_name],
                                '配置前(有调压)PF': pf_pre_with[col_name],
                                '配置前(有调压)有功(MW)': p_pre_with[col_name],
                                '配置前(有调压)无功(MVar)': q_pre_with[col_name],
                                '配置后PF': pf_post[col_name],
                                '配置后有功(MW)': p_post[col_name],
                                '配置后无功(MVar)': q_post[col_name]
                            })

        except Exception as e:
            print(f"    任务一出错: {e}")

        # --- 任务二：小水电节点电压下降百分比 ---
        try:
            # 读取节点电压
            df_vm_pre = pd.read_excel(file_pre_no_reg, sheet_name='节点电压(pu)', index_col=0)
            df_vm_post = pd.read_excel(file_post, sheet_name='节点电压(pu)', index_col=0)

            # 去除重复索引
            df_vm_pre = df_vm_pre[~df_vm_pre.index.duplicated(keep='first')]
            df_vm_post = df_vm_post[~df_vm_post.index.duplicated(keep='first')]

            for node_id in hydro_nodes:
                if node_id in df_vm_pre.index and node_id in df_vm_post.index:
                    v_pre_series = df_vm_pre.loc[node_id]
                    v_post_series = df_vm_post.loc[node_id]

                    for hour in range(24):
                        col_name = f"{hour}点"
                        if col_name in v_pre_series.index:
                            v_pre = v_pre_series[col_name]
                            v_post = v_post_series[col_name]

                            # 计算电压偏差改善率（仅当优化前电压越线时）
                            if v_pre > 1.07 or v_pre < 0.93:
                                dev_pre = abs(v_pre - 1.0)
                                dev_post = abs(v_post - 1.0)
                                if dev_pre != 0:
                                    improve_pct = (dev_pre - dev_post) / dev_pre * 100.0
                                else:
                                    improve_pct = 0.0
                            else:
                                improve_pct = 0.0

                            task2_results.append({
                                '方案': scheme_key,
                                '日期': date_str,
                                '节点ID': node_id,
                                '时刻': hour,
                                '配置前电压(pu)': v_pre,
                                '配置后电压(pu)': v_post,
                                '电压偏差改善率（仅考虑越线情况）(%)': improve_pct
                            })

        except Exception as e:
            print(f"    任务二出错: {e}")

    # 生成DataFrame
    df_task1 = None
    df_task2 = None

    if task1_results:
        df_task1 = pd.DataFrame(task1_results)

        # 计算提升百分比
        df_task1['提升百分比（仅考虑越线情况）(%)'] = (df_task1['配置后PF'] - df_task1['配置前(无调压)PF']) / df_task1['配置前(无调压)PF'] * 100
        df_task1.loc[df_task1['配置前(无调压)PF'] >= 0.95, '提升百分比（仅考虑越线情况）(%)'] = 0
        df_task1['提升百分比（仅考虑越线情况）(%)'] = df_task1['提升百分比（仅考虑越线情况）(%)'].round(2)

    if task2_results:
        df_task2 = pd.DataFrame(task2_results)

    print(f"{scheme_key} 处理完成\n")

    return {'task1': df_task1, 'task2': df_task2}

def main():
    # --- 1. 配置路径与常量 ---
    root_node_id = 1
    hydro_nodes = [26, 15, 44, 22, 11, 7]

    # 输入文件路径
    topology_file = 'data/03_标准化数据/标准格式_支路参数.csv'
    res_pre_no_reg_dir = 'data/优化配置结果/优化前无调压结果/04_潮流计算结果'
    res_pre_with_reg_dir = 'data/优化配置结果/优化前有调压结果/04_潮流计算结果'

    # 输出路径
    output_dir = 'data/潮流计算/多方案对比分析'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 日期列表
    dates = ['10.3', '3.24', '5.7', '7.16', '8.15', '9.27']

    print("\n" + "="*60)
    print("多方案SVG优化配置前后对比分析")
    print("="*60)

    # --- 2. 动态查找方案四的文件夹 ---
    scheme4_folder = find_scheme4_folder()
    if scheme4_folder:
        SCHEMES['方案四']['folder'] = scheme4_folder
        print(f"找到方案四文件夹: {scheme4_folder}")

    # --- 3. 处理所有方案 ---
    all_results = {}

    for scheme_key in ['方案一', '方案二', '方案三', '方案四']:
        scheme_config = SCHEMES[scheme_key]

        result = process_scheme(
            scheme_key=scheme_key,
            scheme_config=scheme_config,
            res_pre_no_reg_dir=res_pre_no_reg_dir,
            res_pre_with_reg_dir=res_pre_with_reg_dir,
            topology_file=topology_file,
            root_node_id=root_node_id,
            hydro_nodes=hydro_nodes,
            dates=dates,
            output_dir=output_dir
        )

        all_results[scheme_key] = result

    # --- 4. 保存各方案的报表 ---
    print("\n" + "="*60)
    print("保存各方案对比分析结果")
    print("="*60)

    for scheme_key, result in all_results.items():
        scheme_name = SCHEMES[scheme_key]['name']

        # 保存任务一结果（功率因数）
        if result['task1'] is not None:
            out_path1 = os.path.join(output_dir, f'对比分析_根节点线路PF_{scheme_name}.csv')
            result['task1'].to_csv(out_path1, index=False, encoding='utf-8-sig')
            print(f"  已保存: {out_path1}")

        # 保存任务二结果（节点电压）
        if result['task2'] is not None:
            out_path2 = os.path.join(output_dir, f'对比分析_小水电节点电压_{scheme_name}.csv')
            result['task2'].to_csv(out_path2, index=False, encoding='utf-8-sig')
            print(f"  已保存: {out_path2}")

    # --- 5. 生成汇总对比表（合并所有方案） ---
    print("\n" + "="*60)
    print("生成多方案汇总对比表")
    print("="*60)

    # 汇总任务一（功率因数）
    task1_all = []
    for scheme_key, result in all_results.items():
        if result['task1'] is not None:
            task1_all.append(result['task1'])

    if task1_all:
        df_task1_combined = pd.concat(task1_all, ignore_index=True)
        out_path_combined1 = os.path.join(output_dir, '对比分析_根节点线路PF_多方案汇总.csv')
        df_task1_combined.to_csv(out_path_combined1, index=False, encoding='utf-8-sig')
        print(f"  已保存: {out_path_combined1}")

    # 汇总任务二（节点电压）
    task2_all = []
    for scheme_key, result in all_results.items():
        if result['task2'] is not None:
            task2_all.append(result['task2'])

    if task2_all:
        df_task2_combined = pd.concat(task2_all, ignore_index=True)
        out_path_combined2 = os.path.join(output_dir, '对比分析_小水电节点电压_多方案汇总.csv')
        df_task2_combined.to_csv(out_path_combined2, index=False, encoding='utf-8-sig')
        print(f"  已保存: {out_path_combined2}")

    # --- 6. 生成方案对比统计表 ---
    print("\n" + "="*60)
    print("生成方案对比统计表")
    print("="*60)

    comparison_stats = []

    for scheme_key, result in all_results.items():
        scheme_name = SCHEMES[scheme_key]['name']

        # 统计功率因数改善情况
        if result['task1'] is not None:
            df_pf = result['task1']
            # 筛选出需要改善的时段（配置前PF < 0.95）
            df_pf_need_improve = df_pf[df_pf['配置前(无调压)PF'] < 0.95]

            if not df_pf_need_improve.empty:
                avg_pf_improve = df_pf_need_improve['提升百分比（仅考虑越线情况）(%)'].mean()
                max_pf_improve = df_pf_need_improve['提升百分比（仅考虑越线情况）(%)'].max()
                min_pf_improve = df_pf_need_improve['提升百分比（仅考虑越线情况）(%)'].min()
                count_pf_need = len(df_pf_need_improve)
            else:
                avg_pf_improve = 0
                max_pf_improve = 0
                min_pf_improve = 0
                count_pf_need = 0
        else:
            avg_pf_improve = 0
            max_pf_improve = 0
            min_pf_improve = 0
            count_pf_need = 0

        # 统计电压改善情况
        if result['task2'] is not None:
            df_v = result['task2']
            # 筛选出需要改善的时段（配置前电压越线）
            df_v_need_improve = df_v[(df_v['配置前电压(pu)'] > 1.07) | (df_v['配置前电压(pu)'] < 0.93)]

            if not df_v_need_improve.empty:
                avg_v_improve = df_v_need_improve['电压偏差改善率（仅考虑越线情况）(%)'].mean()
                max_v_improve = df_v_need_improve['电压偏差改善率（仅考虑越线情况）(%)'].max()
                min_v_improve = df_v_need_improve['电压偏差改善率（仅考虑越线情况）(%)'].min()
                count_v_need = len(df_v_need_improve)
            else:
                avg_v_improve = 0
                max_v_improve = 0
                min_v_improve = 0
                count_v_need = 0
        else:
            avg_v_improve = 0
            max_v_improve = 0
            min_v_improve = 0
            count_v_need = 0

        comparison_stats.append({
            '方案': scheme_key,
            '方案名称': scheme_name,
            'PF需改善时段数': count_pf_need,
            'PF平均改善率(%)': round(avg_pf_improve, 2),
            'PF最大改善率(%)': round(max_pf_improve, 2),
            'PF最小改善率(%)': round(min_pf_improve, 2),
            '电压需改善时段数': count_v_need,
            '电压平均改善率(%)': round(avg_v_improve, 2),
            '电压最大改善率(%)': round(max_v_improve, 2),
            '电压最小改善率(%)': round(min_v_improve, 2)
        })

    if comparison_stats:
        df_comparison = pd.DataFrame(comparison_stats)
        out_path_comparison = os.path.join(output_dir, '方案对比统计表.csv')
        df_comparison.to_csv(out_path_comparison, index=False, encoding='utf-8-sig')
        print(f"  已保存: {out_path_comparison}")
        print("\n方案对比统计预览:")
        print(df_comparison.to_string(index=False))

    print("\n" + "="*60)
    print("所有方案对比分析完成！")
    print("="*60)
    print(f"结果保存在: {output_dir}")

if __name__ == "__main__":
    main()
