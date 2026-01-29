import pandapower as pp
import pandapower.networks
import pandas as pd
import numpy as np
import os
import glob
import warnings

# 忽略部分警告
warnings.filterwarnings("ignore")

def create_network_from_csv(branch_file):
    """
    根据标准化的支路文件建立 Pandapower 电网模型
    """
    net = pp.create_empty_network()
    
    try:
        df_branch = pd.read_csv(branch_file)
    except UnicodeDecodeError:
        df_branch = pd.read_csv(branch_file, encoding='gbk')

    # 1. 创建节点 (Buses)
    all_buses = set(df_branch['from_bus']).union(set(df_branch['to_bus']))
    all_buses.add(1) # 确保平衡节点存在
    
    for bus_id in sorted(list(all_buses)):
        # 设置电压约束范围 0.93 - 1.07 (SVG可双向调节，严格约束)
        pp.create_bus(net, vn_kv=10.0, name=f"Node {bus_id}", index=bus_id,
                      min_vm_pu=0.93, max_vm_pu=1.07)

    # 2. 创建线路 (Lines)
    for _, row in df_branch.iterrows():
        f_bus = int(row['from_bus'])
        t_bus = int(row['to_bus'])
        r_total = row['r_ohm']
        x_total = row['x_ohm']
        c_nf = 10.0 if "电缆" not in str(row['type']) else 300.0
        
        pp.create_line_from_parameters(net, 
                                       from_bus=f_bus, 
                                       to_bus=t_bus, 
                                       length_km=1.0,
                                       r_ohm_per_km=r_total, 
                                       x_ohm_per_km=x_total, 
                                       c_nf_per_km=c_nf, 
                                       max_i_ka=0.4,
                                       name=f"Line {int(row['branch_id'])}")

    # 3. 设置平衡节点 (Slack Bus)
    # 题目要求：根节点电压 1.0
    pp.create_ext_grid(net, bus=1, vm_pu=1.0, name="Slack Bus",
                       min_p_mw=-1000, max_p_mw=1000,
                       min_q_mvar=-1000, max_q_mvar=1000)
    
    return net

def setup_opf_svgs(net, svg_nodes):
    """
    在指定节点添加 SVG (静止无功发生器)
    """
    for bus_id in svg_nodes:
        # 创建 SVG 作为可调无功源
        # p_mw=0 (不发有功)
        # min_q_mvar=-20 (感性/吸收无功)
        # max_q_mvar=20 (容性/发出无功)
        g = pp.create_sgen(net, bus=bus_id, p_mw=0, q_mvar=0,
                          min_p_mw=0, max_p_mw=0,
                          min_q_mvar=-20.0, max_q_mvar=20.0,
                          controllable=True, name=f"SVG_Node_{bus_id}")
        
        # 设置成本函数：最小化 Q^2
        # cp1_mvar * q + cp2_mvar * q^2
        # 我们希望尽可能少用，且分布均匀，所以用二次项
        pp.create_poly_cost(net, element=g, et="sgen", cp1_eur_per_mw=0, cp2_eur_per_mw2=0,
                            cq1_eur_per_mvar=0, cq2_eur_per_mvar2=1.0)

def run_optimization_for_nodes(svg_nodes, scheme_name=""):
    """
    对指定节点运行SVG容量优化

    参数:
        svg_nodes: list, SVG安装节点列表，如 [26, 15, 22]
        scheme_name: str, 方案名称（用于日志显示），如 "方案一_全节点"

    返回:
        dict: {
            'max_q_requirements': {...},  # 各节点最大需求
            'total_capacity_calc': float,  # 总容量（计算值）
            'total_capacity_std': float,   # 总容量（标准化）
            'summary_data': [...],         # 汇总数据
            'results_log': [...]           # 详细运行日志
        }
    """
    print(f"开始执行 SVG 容量优化 (OPF) - {scheme_name}...")

    # 1. 基础配置
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    branch_file = os.path.join(base_dir, "data", "03_标准化数据", "标准格式_支路参数.csv")
    load_files = glob.glob(os.path.join(base_dir, "data", "03_标准化数据", "标准格式_负荷数据_*.csv"))

    # 存储结果
    # 结构: {bus_id: max_q_required}
    max_q_requirements = {bus: 0.0 for bus in svg_nodes}

    # 记录所有时刻的详细结果，用于后续分析
    results_log = []

    # 2. 遍历所有负荷文件
    total_files = len(load_files)
    for f_idx, load_file in enumerate(load_files):
        date_str = os.path.basename(load_file).replace("标准格式_负荷数据_", "").replace(".csv", "")
        print(f"[{f_idx+1}/{total_files}] 处理日期: {date_str}")

        df_load = pd.read_csv(load_file)
        # df_load 结构: bus_id, p_mw_0, q_mvar_0, ...

        for hour in range(24):
            # 构建网络
            net = create_network_from_csv(branch_file)
            setup_opf_svgs(net, svg_nodes)

            # 填充负荷
            p_col = f"p_mw_{hour}"
            q_col = f"q_mvar_{hour}"

            for _, row in df_load.iterrows():
                bus_id = int(row['bus_id'])
                if bus_id == 1: continue

                p_val = row[p_col]
                q_val = row[q_col]

                # 添加负荷 (Pandapower 中 load 正值为消耗)
                pp.create_load(net, bus=bus_id, p_mw=p_val, q_mvar=q_val)

            # 运行 OPF
            try:
                # runopp 使用 AC 最优潮流
                pp.runopp(net, verbose=False)

                if net.OPF_converged:
                    # 提取结果
                    # 查看发电机的无功出力
                    for bus_id in svg_nodes:
                        # 找到连接在该节点的 generator (SVG)
                        # 获取该节点的 gen 索引
                        gen_idx = net.sgen[net.sgen.bus == bus_id].index[0]
                        q_output = net.res_sgen.at[gen_idx, 'q_mvar']

                        # q_output 是发出的无功 (正=容性/发出, 负=感性/吸收)
                        # SVG 可以双向调节，容量取决于最大绝对值

                        q_needed = abs(q_output)

                        if q_needed > max_q_requirements[bus_id]:
                            max_q_requirements[bus_id] = q_needed

                        results_log.append({
                            'date': date_str,
                            'hour': hour,
                            'bus': bus_id,
                            'q_mvar': q_output,
                            'vm_pu': net.res_bus.at[bus_id, 'vm_pu']
                        })
                else:
                    print(f"  警告: {date_str} {hour}时 OPF 未收敛")

            except Exception as e:
                print(f"  错误: {date_str} {hour}时 计算出错: {e}")

    # 3. 计算汇总数据
    print("\n" + "="*60)
    print("优化结果：各节点建议 SVG 容量 (含标准化选型)")
    print("="*60)
    print(f"{'节点ID':<8} {'计算需求 (Mvar)':<15} {'建议选型 (Mvar)':<15} {'选型说明':<15}")
    print("-" * 60)

    total_cap_calc = 0
    total_cap_std = 0
    summary_data = []

    # 设定标准档位步长 (例如 50kvar = 0.05 Mvar)
    step_mvar = 0.05

    for bus_id in svg_nodes:
        cap = max_q_requirements[bus_id]

        # 1. 计算值 (保留3位)
        cap_calc = np.ceil(cap * 1000) / 1000

        # 2. 标准化选型 (向上取整到 0.05 的倍数)
        if cap <= 0:
            cap_std = 0.0
        else:
            cap_std = np.ceil(cap / step_mvar) * step_mvar

        print(f"{bus_id:<8} {cap_calc:<15} {cap_std:<15.2f} {f'Step={step_mvar}':<15}")

        total_cap_calc += cap_calc
        total_cap_std += cap_std

        summary_data.append({
            "节点ID": bus_id,
            "建议选型_Mvar": round(cap_std, 2),
            "计算需求_Mvar": cap_calc,
            "原始精确值_Mvar": cap,
            "选型步长": step_mvar
        })

    print("-" * 60)
    print(f"总容量:   {total_cap_calc:.3f} Mvar (计算) / {total_cap_std:.2f} Mvar (选型)")
    print("="*60)

    # 4. 返回结果
    return {
        'max_q_requirements': max_q_requirements,
        'total_capacity_calc': total_cap_calc,
        'total_capacity_std': total_cap_std,
        'summary_data': summary_data,
        'results_log': results_log
    }

def save_results(result, output_prefix, output_dir):
    """
    保存优化结果到文件

    参数:
        result: dict, run_optimization_for_nodes返回的结果
        output_prefix: str, 输出文件前缀，如 "方案一_多点分散"
        output_dir: str, 输出目录路径

    返回:
        str: 配置文件路径
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 保存配置建议表
    df_summary = pd.DataFrame(result['summary_data'])
    summary_file = os.path.join(output_dir, f"{output_prefix}_SVG优化配置结果.csv")
    df_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"[保存成功] 优化配置结果已保存至: {summary_file}")

    # 2. 保存详细运行日志 (包含每个时刻的出力)
    if result['results_log']:
        df_log = pd.DataFrame(result['results_log'])
        log_file = os.path.join(output_dir, f"{output_prefix}_SVG优化详细日志.csv")
        df_log.to_csv(log_file, index=False, encoding='utf-8-sig')
        print(f"[保存成功] 详细运行日志已保存至: {log_file}")

    return summary_file

def run_power_flow_for_scheme(config_file, scheme_name):
    """
    为指定方案运行潮流计算

    参数:
        config_file: str, SVG配置文件路径
        scheme_name: str, 方案名称，如 "方案一_多点分散"
    """
    print(f"\n{'='*60}")
    print(f"开始计算 {scheme_name} 的潮流结果...")
    print(f"{'='*60}")

    # 导入潮流计算函数
    try:
        # 动态导入10_优化配置潮流计算.py中的函数
        import importlib.util
        script_dir = os.path.dirname(os.path.abspath(__file__))
        power_flow_script = os.path.join(script_dir, "10_优化配置潮流计算.py")

        spec = importlib.util.spec_from_file_location("power_flow_module", power_flow_script)
        if spec is None or spec.loader is None:
            raise ImportError("无法加载潮流计算模块")

        power_flow_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(power_flow_module)

        run_daily_simulation_v2 = power_flow_module.run_daily_simulation_v2

    except Exception as e:
        print(f"[错误] 无法导入潮流计算模块: {e}")
        print(f"[跳过] {scheme_name} 的潮流计算")
        return

    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_folder = os.path.join(base_dir, "data", "03_标准化数据")
    output_folder = os.path.join(base_dir, "data", "潮流计算", f"{scheme_name}_潮流结果")

    branch_file = os.path.join(input_folder, "标准格式_支路参数.csv")
    load_files = glob.glob(os.path.join(input_folder, "标准格式_负荷数据_*.csv"))

    if not os.path.exists(branch_file):
        print(f"[错误] 找不到支路文件: {branch_file}")
        return

    if not os.path.exists(config_file):
        print(f"[错误] 找不到配置文件: {config_file}")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 运行潮流计算
    all_violations = []
    for fpath in load_files:
        fname = os.path.basename(fpath)
        date_str = fname.replace("标准格式_负荷数据_", "").replace(".csv", "")

        try:
            daily_violations = run_daily_simulation_v2(
                branch_file, fpath, date_str, output_folder, config_file
            )
            all_violations.extend(daily_violations)
        except Exception as e:
            print(f"[错误] {date_str} 潮流计算失败: {e}")

    # 保存越限汇总
    violation_path = os.path.join(output_folder, "电压越限汇总.xlsx")
    df_v = pd.DataFrame(all_violations)
    cols = ["日期", "时刻", "节点编号", "电压值", "越限类型"]

    if df_v.empty:
        df_v = pd.DataFrame(columns=cols)
    else:
        existing_cols = [c for c in cols if c in df_v.columns]
        df_v = df_v[existing_cols]

    try:
        df_v.to_excel(violation_path, index=False)
        if all_violations:
            print(f"[提示] 越限汇总已保存: {violation_path}")
        else:
            print(f"[提示] 所有时刻均未出现电压越限")
    except Exception as e:
        print(f"[错误] 保存越限汇总失败: {e}")

    print(f"[完成] {scheme_name} 潮流计算结果已保存至: {output_folder}\n")

def run_scheme_1(output_dir):
    """方案一：多点分散补偿 - 所有5个节点"""
    print("\n" + "="*60)
    print("方案一：多点分散补偿")
    print("配置节点: [26, 15, 22, 11, 7] - 全部5个节点")
    print("="*60)

    svg_nodes = [26, 15, 22, 11, 7]
    result = run_optimization_for_nodes(svg_nodes, "方案一_多点分散")
    config_file = save_results(result, "方案一_多点分散", output_dir)

    # 运行潮流计算
    run_power_flow_for_scheme(config_file, "方案一_多点分散")

    return result

def run_scheme_2(output_dir):
    """方案二：双核心协同补偿 - 东山+红溪"""
    print("\n" + "="*60)
    print("方案二：双核心协同补偿")
    print("配置节点: [26, 15] - 东山水电站 + 红溪支线")
    print("="*60)

    svg_nodes = [26, 15]
    result = run_optimization_for_nodes(svg_nodes, "方案二_双核心")
    config_file = save_results(result, "方案二_双核心", output_dir)

    # 运行潮流计算
    run_power_flow_for_scheme(config_file, "方案二_双核心")

    return result

def run_scheme_3(output_dir):
    """方案三：单核心集中补偿 - 仅东山"""
    print("\n" + "="*60)
    print("方案三：单核心集中补偿")
    print("配置节点: [26] - 仅东山水电站")
    print("="*60)

    svg_nodes = [26]
    result = run_optimization_for_nodes(svg_nodes, "方案三_单核心")
    config_file = save_results(result, "方案三_单核心", output_dir)

    # 运行潮流计算
    run_power_flow_for_scheme(config_file, "方案三_单核心")

    return result

def run_scheme_4(output_dir):
    """方案四：最优双节点优选 - 遍历C(5,2)=10种组合"""
    from itertools import combinations

    print("\n" + "="*60)
    print("方案四：最优双节点优选")
    print("配置方式: 遍历所有双节点组合，选择总容量最小的方案")
    print("="*60)

    # 1. 生成所有双节点组合
    all_nodes = [26, 15, 22, 11, 7]
    all_combinations = list(combinations(all_nodes, 2))

    # 节点名称映射
    node_names = {
        26: "东山水电站",
        15: "红溪支线",
        22: "下严支线",
        11: "细岭电站支线",
        7: "南山电站支线"
    }

    print(f"\n总共需要计算 {len(all_combinations)} 种双节点组合:")
    for i, combo in enumerate(all_combinations):
        names = [node_names[n] for n in combo]
        print(f"  {i+1}. {combo} - {names[0]} + {names[1]}")
    print()

    # 2. 存储所有组合的结果
    comparison_results = []

    # 跟踪最优组合
    best_combo = None
    best_total_capacity = float('inf')
    best_result = None

    # 3. 遍历每个组合
    for combo_idx, combo in enumerate(all_combinations):
        svg_nodes = list(combo)
        combo_name = f"组合{combo_idx+1}_{combo[0]}_{combo[1]}"

        print(f"\n[{combo_idx+1}/{len(all_combinations)}] 计算组合: {combo}")

        # 运行优化
        result = run_optimization_for_nodes(svg_nodes, combo_name)

        # 记录对比数据
        comparison_results.append({
            '组合编号': combo_idx + 1,
            '节点组合': str(combo),
            '节点1': combo[0],
            '节点1名称': node_names[combo[0]],
            '节点1容量_Mvar': result['summary_data'][0]['建议选型_Mvar'],
            '节点2': combo[1],
            '节点2名称': node_names[combo[1]],
            '节点2容量_Mvar': result['summary_data'][1]['建议选型_Mvar'],
            '总容量_标准化_Mvar': result['total_capacity_std'],
            '总容量_计算值_Mvar': round(result['total_capacity_calc'], 3)
        })

        # 更新最优组合
        if result['total_capacity_std'] < best_total_capacity:
            best_total_capacity = result['total_capacity_std']
            best_combo = combo
            best_result = result
            print(f"  >>> 发现新的最优组合！总容量: {best_total_capacity:.2f} Mvar")

    # 4. 保存对比表（所有10个组合）
    df_comparison = pd.DataFrame(comparison_results)
    df_comparison = df_comparison.sort_values('总容量_标准化_Mvar', ascending=True)
    df_comparison.insert(0, '排名', range(1, len(df_comparison) + 1))

    comparison_file = os.path.join(output_dir, "方案四_最优双节点_组合对比表.csv")
    df_comparison.to_csv(comparison_file, index=False, encoding='utf-8-sig')
    print(f"\n[保存成功] 对比表已保存至: {comparison_file}")

    # 5. 检查是否找到最优组合
    if best_combo is None or best_result is None:
        print("\n错误: 未找到最优组合！")
        return {
            'comparison_results': comparison_results,
            'best_result': None,
            'best_combo': None
        }

    # 6. 保存最优组合的详细结果
    best_prefix = f"方案四_最优组合_{best_combo[0]}_{best_combo[1]}"
    best_config_file = save_results(best_result, best_prefix, output_dir)

    # 7. 打印最优结果摘要
    print("\n" + "="*60)
    print("方案四执行完成 - 最优结果摘要")
    print("="*60)
    print(f"最优组合: {best_combo}")
    print(f"节点名称: {node_names[best_combo[0]]} + {node_names[best_combo[1]]}")
    print(f"总容量（标准化）: {best_result['total_capacity_std']:.2f} Mvar")
    print(f"总容量（计算值）: {best_result['total_capacity_calc']:.3f} Mvar")
    print("\n各节点配置:")
    for item in best_result['summary_data']:
        node_name = node_names.get(item['节点ID'], f"节点{item['节点ID']}")
        print(f"  节点 {item['节点ID']} ({node_name}): {item['建议选型_Mvar']:.2f} Mvar")
    print("="*60)

    # 8. 运行最优组合的潮流计算
    run_power_flow_for_scheme(best_config_file, best_prefix)

    return {
        'comparison_results': comparison_results,
        'best_result': best_result,
        'best_combo': best_combo
    }

def generate_summary_table(output_dir):
    """生成所有方案的汇总对比表"""
    print("\n" + "="*60)
    print("正在生成总结果汇总表...")
    
    # 定义方案文件映射
    schemes = {
        "方案一_多点分散": "方案一_多点分散_SVG优化配置结果.csv",
        "方案二_双核心": "方案二_双核心_SVG优化配置结果.csv",
        "方案三_单核心": "方案三_单核心_SVG优化配置结果.csv"
    }
    
    # 查找方案四的文件 (因为文件名包含动态的组合ID)
    scheme4_pattern = os.path.join(output_dir, "方案四_最优组合_*_SVG优化配置结果.csv")
    scheme4_files = glob.glob(scheme4_pattern)
    
    if scheme4_files:
        # 取最新的或者唯一的一个
        scheme4_file = os.path.basename(scheme4_files[0])
        schemes["方案四_最优组合"] = scheme4_file
    else:
        print("[警告] 未找到方案四的结果文件")
    
    # 所有涉及的节点
    all_nodes = [26, 15, 22, 11, 7]
    summary_df = pd.DataFrame(index=all_nodes)
    summary_df.index.name = "节点ID"
    
    # 读取各方案数据
    for scheme_name, filename in schemes.items():
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            # print(f"[警告] 文件不存在: {filename}")
            summary_df[scheme_name] = 0.0
            continue
            
        try:
            df = pd.read_csv(filepath)
            # 设置节点ID为索引
            df.set_index("节点ID", inplace=True)
            # 获取建议选型容量，如果节点不在该方案中，则为NaN (后续填0)
            summary_df[scheme_name] = df["建议选型_Mvar"]
        except Exception as e:
            print(f"[错误] 读取文件 {filename} 失败: {e}")
            summary_df[scheme_name] = 0.0
            
    # 填充NaN为0
    summary_df.fillna(0, inplace=True)
    
    # 添加总计行
    summary_df.loc['总计'] = summary_df.sum()
    
    # 保存汇总表
    output_path = os.path.join(output_dir, "SVG优化配置_总结果表.csv")
    summary_df.to_csv(output_path, encoding='utf-8-sig')
    print(f"总结果表已保存: {output_path}")
    print(summary_df)
    print("="*60)

def main():
    """主函数：依次执行所有4个方案"""
    print("\n" + "="*60)
    print("SVG 容量优化 - 多方案对比分析")
    print("="*60)
    print("待执行方案:")
    print("  方案一: 多点分散补偿 [26, 15, 22, 11, 7]")
    print("  方案二: 双核心协同补偿 [26, 15]")
    print("  方案三: 单核心集中补偿 [26]")
    print("  方案四: 最优双节点优选 (10种组合)")
    print("="*60)

    # 配置输出目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "data", "潮流计算")

    # 依次执行各方案
    try:
        result1 = run_scheme_1(output_dir)
        result2 = run_scheme_2(output_dir)
        result3 = run_scheme_3(output_dir)
        result4 = run_scheme_4(output_dir)

        print("\n" + "="*60)
        print("所有方案执行完成！")
        print("="*60)
        print(f"结果文件已保存至: {output_dir}")
        print("\n方案对比:")
        print(f"  方案一（5节点）: 总容量 {result1['total_capacity_std']:.2f} Mvar")
        print(f"  方案二（2节点）: 总容量 {result2['total_capacity_std']:.2f} Mvar")
        print(f"  方案三（1节点）: 总容量 {result3['total_capacity_std']:.2f} Mvar")
        print(f"  方案四（最优2节点）: 总容量 {result4['best_result']['total_capacity_std']:.2f} Mvar")
        print("="*60)

        # 生成汇总表
        generate_summary_table(output_dir)

    except Exception as e:
        print(f"\n错误: 执行过程中出现异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
