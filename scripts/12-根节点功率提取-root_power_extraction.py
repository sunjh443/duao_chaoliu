import pandas as pd
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

def find_scheme4_folder():
    """动态查找方案四的文件夹（格式：方案四_最优组合_X_Y_潮流结果）"""
    base_dir = 'data/潮流计算'
    pattern = os.path.join(base_dir, '方案四_最优组合_*_*_潮流结果')
    matches = glob.glob(pattern)

    if matches:
        return matches[0].replace('\\', '/')
    else:
        print("  [警告] 未找到方案四的潮流结果文件夹")
        return None

def extract_root_power_for_scheme(scheme_key, scheme_config, res_no_reg_dir, res_with_reg_dir,
                                   topology_file, root_node_id, dates):
    """提取单个方案的根节点功率数据

    参数:
        scheme_key: str, 方案键名
        scheme_config: dict, 方案配置
        res_no_reg_dir: str, 优化前无调压结果目录
        res_with_reg_dir: str, 优化前有调压结果目录
        topology_file: str, 拓扑文件路径
        root_node_id: int, 根节点ID
        dates: list, 日期列表

    返回:
        pd.DataFrame: 提取的功率数据
    """
    scheme_name = scheme_config['name']
    res_opt_dir = scheme_config['folder']

    print(f"\n{'='*60}")
    print(f"处理 {scheme_key} - {scheme_name}")
    print(f"配置说明: {scheme_config['description']}")
    print(f"{'='*60}")

    if res_opt_dir is None:
        print(f"  [跳过] {scheme_key} 的潮流结果文件夹未配置")
        return pd.DataFrame()

    if not os.path.exists(res_opt_dir):
        print(f"  [跳过] 找不到潮流结果文件夹: {res_opt_dir}")
        return pd.DataFrame()

    # 读取拓扑信息
    try:
        df_branch = pd.read_csv(topology_file)
    except UnicodeDecodeError:
        df_branch = pd.read_csv(topology_file, encoding='gbk')

    # 找到连接根节点的线路
    root_lines_df = df_branch[(df_branch['from_bus'] == root_node_id) | (df_branch['to_bus'] == root_node_id)]
    root_line_names = [f"Line {int(bid)}" for bid in root_lines_df['branch_id']]
    print(f"找到连接根节点(Node {root_node_id})的线路: {root_line_names}")

    # 数据容器
    all_data = []

    for date_str in dates:
        print(f"  处理日期: {date_str}")

        file_no_reg = os.path.join(res_no_reg_dir, f"潮流结果_{date_str}.xlsx")
        file_with_reg = os.path.join(res_with_reg_dir, f"潮流结果_{date_str}.xlsx")
        file_opt = os.path.join(res_opt_dir, f"优化后潮流结果_{date_str}.xlsx")

        if not os.path.exists(file_no_reg):
            print(f"    警告: 找不到无调压文件")
            continue
        if not os.path.exists(file_opt):
            print(f"    警告: 找不到优化后文件")
            continue

        try:
            # 辅助函数：读取有功无功
            def read_pq(file_path):
                df_p = pd.read_excel(file_path, sheet_name='线路有功(MW)', index_col='线路名称')
                df_q = pd.read_excel(file_path, sheet_name='线路无功(MVar)', index_col='线路名称')
                df_p = df_p[~df_p.index.duplicated(keep='first')]
                df_q = df_q[~df_q.index.duplicated(keep='first')]
                return df_p, df_q

            # 读取三组数据
            df_p_no, df_q_no = read_pq(file_no_reg)

            # 有调压数据可选
            if os.path.exists(file_with_reg):
                df_p_with, df_q_with = read_pq(file_with_reg)
            else:
                df_p_with = None
                df_q_with = None

            df_p_opt, df_q_opt = read_pq(file_opt)

            for line_name in root_line_names:
                if line_name not in df_p_no.index:
                    print(f"    警告: 线路 {line_name} 不存在于无调压结果中")
                    continue

                for hour in range(24):
                    col_name = f"{hour}点"

                    if col_name in df_p_no.columns:
                        p_no = df_p_no.loc[line_name, col_name]
                        q_no = df_q_no.loc[line_name, col_name]

                        if df_p_with is not None and line_name in df_p_with.index:
                            p_with = df_p_with.loc[line_name, col_name]
                            q_with = df_q_with.loc[line_name, col_name]
                        else:
                            p_with = None
                            q_with = None

                        p_opt = df_p_opt.loc[line_name, col_name] if line_name in df_p_opt.index else None
                        q_opt = df_q_opt.loc[line_name, col_name] if line_name in df_q_opt.index else None

                        all_data.append({
                            "方案": scheme_key,
                            "日期": date_str,
                            "线路名称": line_name,
                            "时刻": hour,
                            "无调压_有功(MW)": p_no,
                            "无调压_无功(MVar)": q_no,
                            "有调压_有功(MW)": p_with,
                            "有调压_无功(MVar)": q_with,
                            "优化后_有功(MW)": p_opt,
                            "优化后_无功(MVar)": q_opt
                        })

        except Exception as e:
            print(f"    错误: {e}")
            import traceback
            traceback.print_exc()

    print(f"{scheme_key} 处理完成\n")

    return pd.DataFrame(all_data)

def main():
    # --- 1. 配置路径与常量 ---
    root_node_id = 1

    # 输入路径
    topology_file = 'data/03_标准化数据/标准格式_支路参数.csv'
    res_no_reg_dir = 'data/优化配置结果/优化前无调压结果/04_潮流计算结果'
    res_with_reg_dir = 'data/优化配置结果/优化前有调压结果/04_潮流计算结果'

    # 输出路径
    output_dir = 'data/潮流计算/多方案根节点功率'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 日期列表
    dates = ['10.3', '3.24', '5.7', '7.16', '8.15', '9.27']

    print("\n" + "="*60)
    print("多方案根节点功率提取")
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

        df_result = extract_root_power_for_scheme(
            scheme_key=scheme_key,
            scheme_config=scheme_config,
            res_no_reg_dir=res_no_reg_dir,
            res_with_reg_dir=res_with_reg_dir,
            topology_file=topology_file,
            root_node_id=root_node_id,
            dates=dates
        )

        all_results[scheme_key] = df_result

    # --- 4. 保存各方案的结果 ---
    print("\n" + "="*60)
    print("保存各方案根节点功率数据")
    print("="*60)

    for scheme_key, df_result in all_results.items():
        scheme_name = SCHEMES[scheme_key]['name']

        if not df_result.empty:
            output_file = os.path.join(output_dir, f'根节点功率对比_{scheme_name}.xlsx')
            df_result.to_excel(output_file, index=False)
            print(f"  已保存: {output_file}")
        else:
            print(f"  {scheme_key} 无数据")

    # --- 5. 生成汇总文件（合并所有方案） ---
    print("\n" + "="*60)
    print("生成多方案汇总文件")
    print("="*60)

    all_data = []
    for scheme_key, df_result in all_results.items():
        if not df_result.empty:
            all_data.append(df_result)

    if all_data:
        df_combined = pd.concat(all_data, ignore_index=True)
        output_combined = os.path.join(output_dir, '根节点功率对比_多方案汇总.xlsx')
        df_combined.to_excel(output_combined, index=False)
        print(f"  已保存: {output_combined}")

    print("\n" + "="*60)
    print("所有方案处理完成！")
    print("="*60)
    print(f"结果保存在: {output_dir}")

if __name__ == "__main__":
    main()
