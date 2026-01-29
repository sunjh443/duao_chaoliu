import pandas as pd
import numpy as np
import os
import glob
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

def process_branches(branch_file):
    print(f"正在处理支路数据 (尊重手动补充值): {branch_file} ...")
    try:
        df = pd.read_csv(branch_file)
    except UnicodeDecodeError:
        df = pd.read_csv(branch_file, encoding='gbk')

    # ==========================================
    # 【修正点】已删除“自动填补前3条”的代码块
    # 现在程序会直接读取您手动填写的数值
    # ==========================================

    # 1. 格式标准化：确保数值列都是数字类型
    # 防止Excel保存时出现文本格式导致的读取错误
    cols_to_numeric = ['线路长度', '电阻_R(欧姆)', '电抗_X(欧姆)']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 确保ID是整数
    cols_to_int = ['序号', '起始序号', '终点序号']
    for col in cols_to_int:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # 2. 列名映射 (映射为 Pandapower 标准变量名)
    rename_map = {
        '序号': 'branch_id',
        '起始序号': 'from_bus',
        '终点序号': 'to_bus',
        '电阻_R(欧姆)': 'r_ohm',
        '电抗_X(欧姆)': 'x_ohm',
        '线路长度': 'length_km',
        '线路类型': 'type'
    }
    
    # 执行重命名
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 3. 筛选最终列
    target_cols = ['branch_id', 'from_bus', 'to_bus', 'r_ohm', 'x_ohm', 'length_km', 'type']
    final_cols = [c for c in target_cols if c in df.columns]
    
    return df[final_cols]

def process_loads(p_file, q_file):
    print(f"正在合并负荷数据...")
    
    # 读取文件 (兼容 CSV 和 Excel)
    def load_file(fpath):
        if fpath.endswith('.xlsx') or fpath.endswith('.xls'):
            return pd.read_excel(fpath)
        try:
            return pd.read_csv(fpath)
        except:
            # 尝试作为 Excel 读取 (针对旧逻辑)
            return pd.read_excel(fpath.replace('.csv', '').replace(' - Sheet1', ''))

    df_p = load_file(p_file)
    df_q = load_file(q_file)

    if df_p is None or df_q is None:
        print("读取负荷文件失败。")
        return None

    # 确保有连接键
    if '序号' not in df_p.columns or '序号' not in df_q.columns:
        print("错误：负荷文件中缺少 '序号' 列")
        return None

    # 合并 P 和 Q
    df_merged = pd.merge(df_p, df_q, on='序号', suffixes=('_p', '_q'), how='inner')

    # ==========================================
    # 单位转换：kW/kvar -> MW/MVar (关键步骤)
    # ==========================================
    result_data = []
    
    for idx, row in df_merged.iterrows():
        bus_id = int(row['序号'])
        
        # 初始字典
        node_data = {'bus_id': bus_id}
        
        # 遍历 0-23 点
        for hour in range(24):
            col_p = f"{hour}点_p"
            col_q = f"{hour}点_q"
            
            p_kw = row.get(col_p, 0)
            q_kvar = row.get(col_q, 0)
            
            # 容错处理
            if pd.isna(p_kw): p_kw = 0
            if pd.isna(q_kvar): q_kvar = 0
            
            # 核心转换: 除以 1000
            node_data[f'p_mw_{hour}'] = p_kw / 1000.0
            node_data[f'q_mvar_{hour}'] = q_kvar / 1000.0
            
        result_data.append(node_data)

    return pd.DataFrame(result_data)

def main():
    # 路径强制定位
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # 定义输入输出目录
    base_data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    
    p_load_dir = os.path.join(base_data_dir, "02_中间数据", "负荷整理")
    q_load_dir = os.path.join(base_data_dir, "02_中间数据", "无功负荷整理")
    branch_data_dir = os.path.join(base_data_dir, "01_源数据")

    output_dir = os.path.join(base_data_dir, "03_标准化数据")

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    # ================= 1. 处理支路 (通用) =================
    branch_input = os.path.join(branch_data_dir, "杜岙配电网_支路数据_含阻抗.csv")
    
    if os.path.exists(branch_input):
        df_branch_std = process_branches(branch_input)
        branch_output = os.path.join(output_dir, "标准格式_支路参数.csv")
        df_branch_std.to_csv(branch_output, index=False, encoding='utf-8-sig')
        print(f">> [成功] 生成: {os.path.basename(branch_output)} (已包含手动补充数据)")
    else:
        print(f"错误：找不到支路文件 {branch_input}")

    # ================= 2. 批处理负荷 (按日期) =================
    # 扫描所有 *负荷整理.xlsx 文件
    p_files = glob.glob(os.path.join(p_load_dir, "*负荷整理.xlsx"))
    
    if not p_files:
        print(f"在 {p_load_dir} 中未找到 '*负荷整理.xlsx' 文件。")
        return

    print(f"\n检测到 {len(p_files)} 个负荷文件，开始批处理...")

    for p_file_path in p_files:
        # 提取文件名和日期前缀
        filename = os.path.basename(p_file_path)
        # 假设文件名格式为 "日期+负荷整理.xlsx"，例如 "3.24负荷整理.xlsx"
        date_prefix = filename.replace("负荷整理.xlsx", "")
        
        print(f"\n>>> 正在处理日期: {date_prefix}")
        
        # 构建对应的无功负荷文件名
        q_filename = f"{date_prefix}无功负荷.xlsx"
        q_file_path = os.path.join(q_load_dir, q_filename)
        
        if not os.path.exists(q_file_path):
            print(f"  跳过：找不到对应的无功文件 {q_filename} 在 {q_load_dir}")
            continue
            
        # 处理负荷
        df_load_std = process_loads(p_file_path, q_file_path)
        
        if df_load_std is not None:
            output_filename = f"标准格式_负荷数据_{date_prefix}.csv"
            output_path = os.path.join(output_dir, output_filename)
            df_load_std.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"  >> [成功] 生成: {output_filename}")

    print("\n所有任务处理完成。")

if __name__ == "__main__":
    main()