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
    # 提取所有出现的节点编号 (from_bus 和 to_bus)
    all_buses = set(df_branch['from_bus']).union(set(df_branch['to_bus']))
    
    # 确保节点1存在（作为平衡节点）
    all_buses.add(1)
    
    # 在 Pandapower 中创建节点
    # index 直接使用实际编号，方便对应
    for bus_id in sorted(list(all_buses)):
        pp.create_bus(net, vn_kv=10.0, name=f"Node {bus_id}", index=bus_id)

    # 2. 创建线路 (Lines)
    for _, row in df_branch.iterrows():
        f_bus = int(row['from_bus'])
        t_bus = int(row['to_bus'])
        
        # 获取阻抗参数 (注意：Pandapower 需要单位长度参数)
        # 技巧：我们将 length_km 设为 1.0，直接把总阻抗填入 r_ohm_per_km
        # 这样可以避免因长度极短导致的除法误差
        r_total = row['r_ohm']
        x_total = row['x_ohm']
        
        # 简单估算电容 (架空线小，电缆大)
        # 这里给一个通用经验值，防止矩阵奇异
        c_nf = 10.0 if "电缆" not in str(row['type']) else 300.0
        
        pp.create_line_from_parameters(net, 
                                       from_bus=f_bus, 
                                       to_bus=t_bus, 
                                       length_km=1.0,  # 设为1，阻抗即为总值
                                       r_ohm_per_km=r_total, 
                                       x_ohm_per_km=x_total, 
                                       c_nf_per_km=c_nf, 
                                       max_i_ka=0.4, # 假设载流量
                                       name=f"Line {int(row['branch_id'])}")

    # 3. 设置平衡节点 (Slack Bus)
    # 假设 节点1 是 10kV 母线/发电机
    pp.create_ext_grid(net, bus=1, vm_pu=1.05, name="Slack Bus")
    
    return net

def calculate_power_factor(p_mw, q_mvar):
    """计算功率因数"""
    s_mva = np.sqrt(p_mw**2 + q_mvar**2)
    if s_mva == 0:
        return 1.0
    return np.abs(p_mw) / s_mva

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

def run_daily_simulation(net, load_file, date_str, output_folder, strategy=2):
    """
    运行单日的 24小时 潮流计算
    """
    print(f"  >>> 正在计算日期: {date_str} (策略: {strategy}) ...")
    
    try:
        df_load = pd.read_csv(load_file)
    except:
        df_load = pd.read_excel(load_file)

    # 准备结果容器
    results_vm = {}  # 电压
    results_pf = {}  # 功率因数
    results_line_p = {} # 线路有功
    results_line_q = {} # 线路无功
    results_line_loading = {} # 线路负载率
    results_log = {} # 计算日志
    
    # 获取网络中所有的 Load 索引 (如果有预设的话)
    # 但我们这里选择动态创建 Load，每小时刷新一次更稳健
    
    # 循环 24 小时
    last_converged = False # 记录上一时刻是否收敛，用于热启动

    for hour in range(24):
        # 1. 清除上一时刻的负荷
        net.load.drop(net.load.index, inplace=True)
        
        # 2. 加载当前时刻负荷
        col_p = f"p_mw_{hour}"
        col_q = f"q_mvar_{hour}"
        
        # 遍历负荷文件中的每一个节点
        for _, row in df_load.iterrows():
            bus_id = int(row['bus_id'])
            p_val = row.get(col_p, 0)
            q_val = row.get(col_q, 0)
            
            # 只有当功率不为0时才添加负荷
            if abs(p_val) > 0 or abs(q_val) > 0:
                # 检查节点是否存在 (防止负荷表里有野节点)
                if bus_id in net.bus.index:
                    pp.create_load(net, bus=bus_id, p_mw=p_val, q_mvar=q_val)
        
        # 3. 运行潮流计算
        # 策略 1: 增加迭代次数 (max_iteration=50)
        # 策略 2: 热启动 (init="results")
        
        # 初始计算
        try:
            if last_converged:
                # 尝试热启动
                try:
                    pp.runpp(net, max_iteration=50, init="results")
                    converged = True
                    msg = "收敛 (热启动)"
                except:
                    # 热启动失败，回退到 DC 初始化
                    pp.runpp(net, max_iteration=50, init="dc")
                    converged = True
                    msg = "收敛 (热启动失败->DC启动)"
            else:
                # 上一时刻未收敛，直接使用 DC 初始化
                pp.runpp(net, max_iteration=50, init="dc")
                converged = True
                msg = "收敛 (DC启动)"
                
        except Exception as e:
            converged = False
            msg = f"不收敛: {str(e)}"
            # print(f"    警告: {hour}点 潮流计算不收敛: {msg}")
        
        # --- 电压控制策略循环 ---
        if converged:
            # 最多尝试调整 5 次
            for control_step in range(5):
                if adjust_slack_bus_voltage(net, strategy=strategy):
                    # 如果调整了电压，需要重新计算潮流
                    try:
                        pp.runpp(net, max_iteration=50, init="results")
                        msg += f" -> 调压({control_step+1})"
                    except Exception as e:
                        converged = False
                        msg += f" -> 调压失败: {str(e)}"
                        break
                else:
                    # 不需要调整，跳出循环
                    break
        # -----------------------

        last_converged = converged
        results_log[f"{hour}点"] = msg

        # 4. 提取结果
        if converged:
            # 提取节点电压 (vm_pu)
            # Series转Dict: {bus_id: voltage}
            results_vm[f"{hour}点"] = net.res_bus['vm_pu'].copy()
            
            # 提取/计算 节点功率因数
            # 注意：res_bus 中的 p_mw 是注入功率，正值为发，负值为用
            # 我们主要关心电压，PF计算仅供参考
            p_res = net.res_bus['p_mw']
            q_res = net.res_bus['q_mvar']
            pf_series = pd.Series(index=net.res_bus.index, dtype=float)
            for idx in net.res_bus.index:
                pf_series[idx] = calculate_power_factor(p_res[idx], q_res[idx])
            results_pf[f"{hour}点"] = pf_series
            
            # 提取线路结果
            results_line_p[f"{hour}点"] = net.res_line['p_from_mw'].copy()
            results_line_q[f"{hour}点"] = net.res_line['q_from_mvar'].copy()
            results_line_loading[f"{hour}点"] = net.res_line['loading_percent'].copy()
            
        else:
            # 不收敛填 NaN
            results_vm[f"{hour}点"] = np.nan
            results_pf[f"{hour}点"] = np.nan
            results_line_p[f"{hour}点"] = np.nan
            results_line_q[f"{hour}点"] = np.nan
            results_line_loading[f"{hour}点"] = np.nan

    # --- 整理并保存结果 ---
    # 转为 DataFrame
    df_res_vm = pd.DataFrame(results_vm)
    df_res_pf = pd.DataFrame(results_pf)
    df_res_line_p = pd.DataFrame(results_line_p)
    df_res_line_q = pd.DataFrame(results_line_q)
    df_res_line_loading = pd.DataFrame(results_line_loading)
    
    # 为线路结果增加名称列 (方便识别)
    if not net.line.empty:
        # 确保索引对齐
        line_names = net.line['name']
        # 插入到第一列
        df_res_line_p.insert(0, '线路名称', line_names)
        df_res_line_q.insert(0, '线路名称', line_names)
        df_res_line_loading.insert(0, '线路名称', line_names)

    # 修改线路索引从1开始
    df_res_line_p.index = df_res_line_p.index + 1
    df_res_line_q.index = df_res_line_q.index + 1
    df_res_line_loading.index = df_res_line_loading.index + 1

    df_log = pd.DataFrame.from_dict(results_log, orient='index', columns=['计算状态'])
    
    # 增加节点名称列
    df_res_vm.index.name = '节点编号'
    df_res_pf.index.name = '节点编号'
    df_res_line_p.index.name = '线路索引'
    df_res_line_q.index.name = '线路索引'
    df_res_line_loading.index.name = '线路索引'
    df_log.index.name = '时刻'
    
    # --- 统计越限情况 ---
    violations = []
    v_upper = 1.07
    v_lower = 0.93
    
    for col in df_res_vm.columns:
        s = df_res_vm[col].dropna()
        
        # 越上限
        over_limit = s[s > v_upper]
        for bus_id, val in over_limit.items():
            violations.append({
                "日期": date_str,
                "时刻": col,
                "节点编号": bus_id,
                "电压值": val,
                "越限类型": "偏高 (>1.07)"
            })
            
        # 越下限
        under_limit = s[s < v_lower]
        for bus_id, val in under_limit.items():
            violations.append({
                "日期": date_str,
                "时刻": col,
                "节点编号": bus_id,
                "电压值": val,
                "越限类型": "偏低 (<0.93)"
            })
    # -------------------
    
    # 保存到 Excel (一个日期一个文件)
    output_path = os.path.join(output_folder, f"潮流结果_{date_str}.xlsx")
    with pd.ExcelWriter(output_path) as writer:
        df_res_vm.to_excel(writer, sheet_name="节点电压(pu)")
        df_res_pf.to_excel(writer, sheet_name="节点功率因数")
        df_res_line_p.to_excel(writer, sheet_name="线路有功(MW)")
        df_res_line_q.to_excel(writer, sheet_name="线路无功(MVar)")
        df_res_line_loading.to_excel(writer, sheet_name="线路负载率(%)")
        df_log.to_excel(writer, sheet_name="计算日志")
        
    print(f"  [成功] 结果已生成: {output_path}")
    return violations

def main():
    # 路径配置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    base_data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    input_folder = os.path.join(base_data_dir, "03_标准化数据")
    output_folder = os.path.join(base_data_dir, "04_潮流计算结果")
    
    branch_file = os.path.join(input_folder, "标准格式_支路参数.csv")
    load_folder = input_folder # 负荷文件也在同一个文件夹
    
    # 1. 检查必要文件
    if not os.path.exists(branch_file):
        print(f"错误：找不到支路模型文件 [{branch_file}]")
        return

    # 2. 建立网络模型 (拓扑是一样的，建一次即可)
    print("正在构建电网拓扑模型...")
    net_base = create_network_from_csv(branch_file)
    print(f"模型构建完成: {len(net_base.bus)} 个节点, {len(net_base.line)} 条线路")

    # 3. 扫描负荷文件
    if not os.path.exists(load_folder):
        print(f"错误：找不到负荷文件夹 [{load_folder}]")
        return
    
    # 假设文件名为 "标准格式_负荷数据_日期.csv"
    load_files = glob.glob(os.path.join(load_folder, "标准格式_负荷数据_*.csv"))
    
    if not load_files:
        print(f"未找到负荷数据文件 (格式: 标准格式_负荷数据_*.csv) 在 {load_folder}")
        return

    # 4. 准备输出目录
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # 配置策略: 1=固定1.0 (无调压), 2=动态调整 (有调压)
    strategies = [1]
    
    for strategy in strategies:
        strategy_name = "无调压" if strategy == 1 else "有调压"
        print(f"\n==================================================")
        print(f"开始执行策略: 方案 {strategy} ({strategy_name})")
        
        # 为每个策略创建独立的输出文件夹
        # 假设 output_folder 是 "data/04_潮流计算结果"
        # 我们将其修改为 "data/优化配置结果/优化前{strategy_name}结果/04_潮流计算结果" 以匹配之前的路径结构
        # 或者简单点，直接在 output_folder 下分子文件夹
        
        # 根据用户之前的路径习惯：
        # data/优化配置结果/优化前无调压结果/04_潮流计算结果
        # data/优化配置结果/优化前有调压结果/04_潮流计算结果
        
        # 这里我们修改 output_folder 的指向
        # 注意：base_data_dir 是 data/
        current_output_folder = os.path.join(base_data_dir, "优化配置结果", f"优化前{strategy_name}结果", "04_潮流计算结果")
        
        if not os.path.exists(current_output_folder):
            os.makedirs(current_output_folder)
            print(f"创建输出目录: {current_output_folder}")

        # 5. 循环计算每一天
        all_violations = []
        for fpath in load_files:
            # 提取日期标识 (例如从文件名中提取 "3.24")
            fname = os.path.basename(fpath)
            # 文件名格式: 标准格式_负荷数据_3.24.csv
            date_str = fname.replace("标准格式_负荷数据_", "").replace(".csv", "")
            
            # 深拷贝网络，防止不同日期的计算互相污染
            import copy
            net_daily = copy.deepcopy(net_base)
            
            # 运行
            daily_violations = run_daily_simulation(net_daily, fpath, date_str, current_output_folder, strategy=strategy)
            all_violations.extend(daily_violations)

        # 保存汇总越限结果
        if all_violations:
            violation_path = os.path.join(current_output_folder, "越线结果汇总.xlsx")
            df_v = pd.DataFrame(all_violations)
            # 调整列顺序
            cols = ["日期", "时刻", "节点编号", "电压值", "越限类型"]
            # 确保列存在 (防止空数据时报错)
            if not df_v.empty:
                 # 仅保留存在的列
                 existing_cols = [c for c in cols if c in df_v.columns]
                 df_v = df_v[existing_cols]
            df_v.to_excel(violation_path, index=False)
            print(f"\n[提示] {strategy_name} 越线结果汇总已保存: {violation_path}")
        else:
            print(f"\n[提示] {strategy_name} 所有计算结果均未出现电压越限。")

    print("\n--------------------------------")
    print(f"所有计算完成！请查看 [{output_folder}] 文件夹。")

if __name__ == "__main__":
    main()