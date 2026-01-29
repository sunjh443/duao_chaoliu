import pandapower as pp
import pandapower.topology
import pandas as pd
import numpy as np
import os
import glob
import warnings

# 忽略计算中的收敛性警告（只在最后统计）
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
    all_buses.add(1)
    
    for bus_id in sorted(list(all_buses)):
        # 设置电压约束范围 (OPF需要)
        # 上限 1.07 (硬约束)
        # 下限 0.93 (放宽，避免低电压导致不收敛，因为电抗器只解决高电压)
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
    # 根节点电压固定为 1.0
    pp.create_ext_grid(net, bus=1, vm_pu=1.0, name="Slack Bus",
                       min_p_mw=-1000, max_p_mw=1000,
                       min_q_mvar=-1000, max_q_mvar=1000)
    
    return net

def setup_configured_svgs(net, config_file):
    """
    根据配置文件安装指定容量的 SVG
    根据配置文件安装指定容量的 SVG (静止无功发生器)
    """
    if not os.path.exists(config_file):
        print(f"警告: 配置文件未找到 {config_file}")
        return
        
    df_config = pd.read_csv(config_file)
    
    for _, row in df_config.iterrows():
        bus_id = int(row['节点ID'])
        capacity_mvar = float(row['建议选型_Mvar'])
        
        if capacity_mvar <= 0:
            continue
            
        # 创建发电机作为可调无功源 (OPF模型)
        # min_q_mvar = -capacity_mvar (最大吸收能力)
        # max_q_mvar = capacity_mvar (最大发出能力，SVG可双向)
        # 注意：如果只允许吸收，max_q_mvar 设为 0
        g = pp.create_sgen(net, bus=bus_id, p_mw=0, q_mvar=0,
                          min_p_mw=0, max_p_mw=0,
                          min_q_mvar=-capacity_mvar, max_q_mvar=capacity_mvar,
                          controllable=True, name=f"SVG_Node_{bus_id}")
        
        # 设置成本函数：最小化 Q^2 (平滑出力)
        pp.create_poly_cost(net, element=g, et="sgen", cp1_eur_per_mw=0, cp2_eur_per_mw2=0,
                            cq1_eur_per_mvar=0, cq2_eur_per_mvar2=1.0)

def calculate_power_factor(p_mw, q_mvar):
    """计算功率因数"""
    s_mva = np.sqrt(p_mw**2 + q_mvar**2)
    if s_mva == 0:
        return 1.0
    return np.abs(p_mw) / s_mva

def heuristic_voltage_control(net):
    """
    简单的启发式电压控制：
    如果电压偏高，增加 SVG 出力（吸收无功）。
    """
    # 1. 找到所有可调 SVG
    # 注意：在 run_daily_simulation_v2 中，我们可能把 controllable 设为了 False
    # 这里需要重新识别，或者假设 sgen 都是 SVG (根据 setup_configured_svgs)
    svgs = net.sgen.index
    if len(svgs) == 0:
        try:
            pp.runpp(net)
            return True, "无SVG"
        except Exception as e:
            return False, f"计算失败: {str(e)}"
        
    # 确保是 PQ 模式
    net.sgen['controllable'] = False
    
    # 确保 q_mvar 列存在且无 NaN (防止 OPF 失败后该列丢失或未初始化)
    if 'q_mvar' not in net.sgen.columns:
        net.sgen['q_mvar'] = 0.0
    else:
        net.sgen['q_mvar'] = net.sgen['q_mvar'].fillna(0.0)
    
    # 2. 迭代调整
    max_iter = 10
    
    for i in range(max_iter):
        try:
            pp.runpp(net, max_iteration=50)
        except pp.LoadflowNotConverged:
            return False, "启发式过程潮流不收敛"
            
        # 检查电压
        if net.res_bus.empty:
            return False, "无结果"
            
        vm = net.res_bus['vm_pu']
        # 只需要关注安装了 SVG 的节点或全网最高电压
        max_v = vm.max()
        min_v = vm.min()
        
        # 获取各 SVG 出力详情
        svg_details = []
        for idx in svgs:
            bus_id = net.sgen.at[idx, 'bus']
            q_val = net.sgen.at[idx, 'q_mvar']
            svg_details.append(f"N{bus_id}:{q_val:.2f}")
        details_str = ", ".join(svg_details)
        
        # 目标：将电压控制在 0.95 - 1.05 之间
        if max_v <= 1.05 and min_v >= 0.95: 
            return True, f"启发式收敛(iter={i}, Vmax={max_v:.4f}, Vmin={min_v:.4f}, {details_str})"
            
        # 调整
        changed = False
        for idx in svgs:
            bus = net.sgen.at[idx, 'bus']
            if bus not in vm.index: continue
            v_node = vm.at[bus]
            
            current_q = net.sgen.at[idx, 'q_mvar']
            min_q = net.sgen.at[idx, 'min_q_mvar'] # 负值 (最大吸收)
            max_q = net.sgen.at[idx, 'max_q_mvar'] # 正值 (最大发出)
            
            # 步长：容量的 20%
            capacity = max(abs(min_q), abs(max_q))
            step = capacity * 0.2
            
            # 策略：
            # 1. 电压偏高 -> 吸收无功 (减少 Q)
            if v_node > 1.03 or max_v > 1.05:
                if current_q > min_q + 1e-4:
                    new_q = max(current_q - step, min_q)
                    if abs(new_q - current_q) > 1e-4:
                        net.sgen.at[idx, 'q_mvar'] = new_q
                        changed = True
                        
            # 2. 电压偏低 -> 发出无功 (增加 Q)
            elif v_node < 0.97 or min_v < 0.95:
                if current_q < max_q - 1e-4:
                    new_q = min(current_q + step, max_q)
                    if abs(new_q - current_q) > 1e-4:
                        net.sgen.at[idx, 'q_mvar'] = new_q
                        changed = True
        
        if not changed:
            return True, f"启发式结束(已尽力, Vmax={max_v:.4f}, Vmin={min_v:.4f}, {details_str})"
            
    # 最后一次计算结果
    svg_details = []
    for idx in svgs:
        bus_id = net.sgen.at[idx, 'bus']
        q_val = net.sgen.at[idx, 'q_mvar']
        svg_details.append(f"N{bus_id}:{q_val:.2f}")
    details_str = ", ".join(svg_details)
    
    return True, f"启发式完成(达最大迭代, Vmax={vm.max():.4f}, {details_str})"

def adjust_slack_bus_voltage(net, strategy=2, v_max_limit=1.07, v_min_limit=0.93):
    """
    电压控制策略框架：
    根据当前计算得到的节点电压，调整平衡节点(Slack Bus)的电压设定值。
    
    参数:
        strategy (int): 
            1 - 固定平衡节点电压为 1.0 pu
            2 - 动态调整 (根据系统最高/最低电压)
            
    返回:
        bool: 如果进行了调整返回 True，否则返回 False
    """
    # 获取当前平衡节点的电压设定值
    # 假设只有一个 ext_grid
    if net.ext_grid.empty:
        return False
        
    slack_idx = net.ext_grid.index[0]
    current_slack_vm = net.ext_grid.at[slack_idx, 'vm_pu']
    new_slack_vm = current_slack_vm

    if strategy == 1:
        # 方案1: 强制将平衡节点电压设为 1.0
        target_vm = 1.0
        if abs(current_slack_vm - target_vm) > 1e-4:
            net.ext_grid.at[slack_idx, 'vm_pu'] = target_vm
            return True
        return False

    elif strategy == 2:
        # 方案2: 现有策略 (根据电压极值调整)
        
        # 获取当前所有节点的电压结果 (排除 NaN)
        if net.res_bus.empty or 'vm_pu' not in net.res_bus:
            return False
            
        vm_pu = net.res_bus['vm_pu'].dropna()
        if vm_pu.empty:
            return False
            
        v_max = vm_pu.max()
        v_min = vm_pu.min()
        
        # --- 策略逻辑 ---
        # 如果系统最高电压超过上限，降低平衡节点电压
        if v_max > v_max_limit:
            # 简单的步长调整，每次降低 0.005 pu
            new_slack_vm -= 0.005 
            
        # 如果系统最低电压低于下限，提高平衡节点电压
        # (注意：如果同时满足高电压和低电压越限，优先处理高电压越限，或者根据具体需求调整)
        elif v_min < v_min_limit:
            new_slack_vm += 0.005
            
        # 限制平衡节点电压的调节范围 (例如 0.95 - 1.05)
        new_slack_vm = max(0.95, min(1.05, new_slack_vm))
        
        # 检查是否有实质性变化
        if abs(new_slack_vm - current_slack_vm) > 1e-4:
            net.ext_grid.at[slack_idx, 'vm_pu'] = new_slack_vm
            return True
            
        return False
        
    return False

def run_daily_simulation_v2(branch_file, load_file, date_str, output_folder, config_file):
    print(f"  >>> 正在计算日期: {date_str} ...")
    
    try:
        df_load = pd.read_csv(load_file)
    except:
        df_load = pd.read_excel(load_file)

    results_vm = {}
    results_pf = {}
    results_line_p = {}
    results_line_q = {}
    results_line_loading = {}
    results_log = {}
    
    # 专门记录 SVG 出力
    svg_outputs = {} 

    for hour in range(24):
        # 1. 重建网络
        net = create_network_from_csv(branch_file)
        
        # 2. 安装 SVG
        setup_configured_svgs(net, config_file)
        
        # 3. 加载负荷
        col_p = f"p_mw_{hour}"
        col_q = f"q_mvar_{hour}"
        
        for _, row in df_load.iterrows():
            bus_id = int(row['bus_id'])
            if bus_id == 1: continue
            p_val = row.get(col_p, 0)
            q_val = row.get(col_q, 0)
            if abs(p_val) > 0 or abs(q_val) > 0:
                pp.create_load(net, bus=bus_id, p_mw=p_val, q_mvar=q_val)
        
        # 4. 运行计算
        # 定义单次计算逻辑 (PF check -> OPF -> Fallback Heuristic -> Fallback PF)
        def run_calc_once(net):
            msg = ""
            converged = False
            
            # --- 步骤 0: 预检查 (Pre-check) ---
            # 先尝试普通潮流。如果当前状态(SVG为0)已经满足电压要求，
            # 则不需要运行不稳定的 OPF。
            try:
                net.sgen['controllable'] = False # 暂时视为普通负载
                pp.runpp(net, verbose=False)
                
                # 检查电压越限
                if not net.res_bus.empty:
                    v_max = net.res_bus['vm_pu'].max()
                    v_min = net.res_bus['vm_pu'].min()
                    # 如果电压在安全范围内 (0.93 - 1.07)
                    if v_max <= 1.07 and v_min >= 0.93:
                        return True, f"PF收敛(无需优化, Vmax={v_max:.4f})"
            except:
                # 如果PF都算不过去，说明问题比较大，交给后面处理
                pass

            # --- 步骤 1: 尝试 OPF ---
            try:
                # 恢复 OPF 设置
                net.sgen['controllable'] = True
                # 使用 warm start (利用刚才 PF 的结果作为初值，提高收敛率)
                pp.runopp(net, verbose=False, init="results") 
                
                if net.OPF_converged:
                    converged = True
                    msg = "OPF收敛"
                    return converged, msg
                else:
                    raise Exception("OPF未收敛")
            except (Exception, KeyboardInterrupt) as e:
                # 捕获 KeyboardInterrupt 允许用户跳过卡住的 OPF
                err_type = "用户中断" if isinstance(e, KeyboardInterrupt) else "OPF失败"
                # print(f"    {err_type}: {str(e)} -> 切换到启发式控制")
                
                # --- 步骤 2: 尝试启发式控制 ---
                try:
                    converged, h_msg = heuristic_voltage_control(net)
                    msg = f"{err_type}->{h_msg}"
                    if converged:
                        return converged, msg
                except Exception as e2:
                    msg = f"{err_type}->启发式出错({str(e2)})"

                # --- 步骤 3: 最后的保底: 普通潮流 ---
                try:
                    net.sgen['controllable'] = False
                    # net.sgen['q_mvar'] = 0.0 # 不强制归零，保留启发式的结果
                    pp.runpp(net, max_iteration=50, init="dc")
                    converged = True
                    msg += "->PF收敛"
                except Exception as e3:
                    converged = False
                    msg += f"->PF失败({str(e3)})"
                    
            return converged, msg

        # 初始计算
        converged, msg = run_calc_once(net)
        
        # 电压控制循环
        if converged:
             # 最多尝试调整 5 次
             for control_step in range(5):
                if adjust_slack_bus_voltage(net, strategy=2):
                     # 如果调整了电压，需要重新计算潮流
                     converged, new_msg = run_calc_once(net)
                     msg += f" -> 调压({control_step+1}):{new_msg}"
                     if not converged:
                         break
                else:
                    # 不需要调整，跳出循环
                    break

        results_log[f"{hour}点"] = msg

        # 5. 提取结果
        if converged:
            results_vm[f"{hour}点"] = net.res_bus['vm_pu'].copy()
            
            p_res = net.res_bus['p_mw']
            q_res = net.res_bus['q_mvar']
            pf_series = pd.Series(index=net.res_bus.index, dtype=float)
            for idx in net.res_bus.index:
                pf_series[idx] = calculate_power_factor(p_res[idx], q_res[idx])
            results_pf[f"{hour}点"] = pf_series
            
            results_line_p[f"{hour}点"] = net.res_line['p_from_mw'].copy()
            results_line_q[f"{hour}点"] = net.res_line['q_from_mvar'].copy()
            results_line_loading[f"{hour}点"] = net.res_line['loading_percent'].copy()
            
            # 提取 SVG 出力
            for idx, gen in net.sgen.iterrows():
                if "SVG" in gen['name']:
                    name = gen['name'].replace("SVG_Node_", "Node ")
                    if name not in svg_outputs:
                        svg_outputs[name] = {}
                    # 记录吸收的无功 (正值)
                    q_val = -net.res_sgen.at[idx, 'q_mvar']
                    svg_outputs[name][f"{hour}点"] = round(q_val, 4)
            
        else:
            results_vm[f"{hour}点"] = np.nan
            results_pf[f"{hour}点"] = np.nan
            results_line_p[f"{hour}点"] = np.nan
            results_line_q[f"{hour}点"] = np.nan
            results_line_loading[f"{hour}点"] = np.nan

    # --- 整理并保存结果 ---
    df_res_vm = pd.DataFrame(results_vm)
    df_res_pf = pd.DataFrame(results_pf)
    df_res_line_p = pd.DataFrame(results_line_p)
    df_res_line_q = pd.DataFrame(results_line_q)
    df_res_line_loading = pd.DataFrame(results_line_loading)
    df_res_svg = pd.DataFrame(svg_outputs)
    
    if not net.line.empty:
        line_names = net.line['name']
        df_res_line_p.insert(0, '线路名称', line_names)
        df_res_line_q.insert(0, '线路名称', line_names)
        df_res_line_loading.insert(0, '线路名称', line_names)

    df_res_line_p.index = df_res_line_p.index + 1
    df_res_line_q.index = df_res_line_q.index + 1
    df_res_line_loading.index = df_res_line_loading.index + 1

    df_log = pd.DataFrame.from_dict(results_log, orient='index', columns=['计算状态'])
    
    df_res_vm.index.name = '节点编号'
    df_res_pf.index.name = '节点编号'
    df_res_line_p.index.name = '线路索引'
    df_res_line_q.index.name = '线路索引'
    df_res_line_loading.index.name = '线路索引'
    df_log.index.name = '时刻'
    
    # 统计越限
    violations = []
    v_upper = 1.07
    v_lower = 0.93
    
    for col in df_res_vm.columns:
        s = df_res_vm[col].dropna()
        over_limit = s[s > v_upper]
        for bus_id, val in over_limit.items():
            violations.append({
                "日期": date_str,
                "时刻": col,
                "节点编号": bus_id,
                "电压值": val,
                "越限类型": "偏高 (>1.07)"
            })
        under_limit = s[s < v_lower]
        for bus_id, val in under_limit.items():
            violations.append({
                "日期": date_str,
                "时刻": col,
                "节点编号": bus_id,
                "电压值": val,
                "越限类型": "偏低 (<0.93)"
            })
    
    output_path = os.path.join(output_folder, f"优化后潮流结果_{date_str}.xlsx")
    with pd.ExcelWriter(output_path) as writer:
        df_res_vm.to_excel(writer, sheet_name="节点电压(pu)")
        df_res_pf.to_excel(writer, sheet_name="节点功率因数")
        df_res_line_p.to_excel(writer, sheet_name="线路有功(MW)")
        df_res_line_q.to_excel(writer, sheet_name="线路无功(MVar)")
        df_res_line_loading.to_excel(writer, sheet_name="线路负载率(%)")
        if not df_res_svg.empty:
            df_res_svg.to_excel(writer, sheet_name="SVG出力(MVar)")
        df_log.to_excel(writer, sheet_name="计算日志")
        
    print(f"  [成功] 结果已生成: {output_path}")
    return violations

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    input_folder = os.path.join(base_data_dir, "03_标准化数据")
    
    # 新的输出文件夹
    output_folder = os.path.join(base_data_dir, "优化配置结果", "06_优化后潮流计算")
    
    branch_file = os.path.join(input_folder, "标准格式_支路参数.csv")
    config_file = os.path.join(os.path.dirname(script_dir), "潮流计算", "SVG优化配置结果.csv")
    
    if not os.path.exists(branch_file):
        print(f"错误：找不到支路模型文件 [{branch_file}]")
        return
    if not os.path.exists(config_file):
        print(f"错误：找不到配置结果文件 [{config_file}]")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    load_files = glob.glob(os.path.join(input_folder, "标准格式_负荷数据_*.csv"))
    
    all_violations = []
    for fpath in load_files:
        fname = os.path.basename(fpath)
        date_str = fname.replace("标准格式_负荷数据_", "").replace(".csv", "")
        
        daily_violations = run_daily_simulation_v2(branch_file, fpath, date_str, output_folder, config_file)
        all_violations.extend(daily_violations)

    violation_path = os.path.join(output_folder, "优化后越线结果汇总.xlsx")
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
            print(f"\n[提示] 越线结果汇总已保存: {violation_path}")
        else:
            print(f"\n[提示] 所有计算结果均未出现电压越限。已生成空汇总表: {violation_path}")
    except PermissionError:
        print(f"\n[错误] 无法写入越线汇总文件: {violation_path}。请检查文件是否被占用。")
    except Exception as e:
        print(f"\n[错误] 写入越线汇总文件失败: {str(e)}")

    print("\n--------------------------------")
    print(f"所有计算完成！请查看 [{output_folder}] 文件夹。")

if __name__ == "__main__":
    main()
