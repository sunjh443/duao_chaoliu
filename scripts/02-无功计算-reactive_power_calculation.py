import pandas as pd
import numpy as np
import os
import glob
import warnings
import sys

# 添加当前目录到 sys.path 以便导入同级模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
try:
    import clean_hydro_duplicates
except ImportError:
    print("Warning: Could not import clean_hydro_duplicates.py")

warnings.filterwarnings("ignore")

def clean_id_column(series):
    return pd.to_numeric(series, errors='coerce').fillna(0).astype(int).astype(str)

def calculate_q(p_val, pf_val):
    """
    计算无功功率 Q = P × tan(arccos(PF))

    修改说明：
    - 只过滤明显错误的PF值（>1.01或<0.01）
    - 对PF≈1的情况，使用0.999避免Q完全为0（因tan(0)=0）
    """
    # 只过滤明显错误的PF值
    pf_clean = np.where((pf_val > 1.01) | (pf_val < 0.01), np.nan, pf_val)

    # 对PF≈1的情况，限制在0.999以下，避免tan(0)=0导致Q=0
    pf_clean = np.where(pf_clean > 0.999, 0.999, pf_clean)

    tan_phi = np.tan(np.arccos(pf_clean))
    q_val = p_val * tan_phi
    return np.nan_to_num(q_val, nan=0.0)

def main():
    # 路径定位
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # ================= 配置区域 =================
    base_data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    source_dir = os.path.join(base_data_dir, '01_源数据') # 源数据文件夹
    p_source_dir = os.path.join(base_data_dir, '02_中间数据', '有功负荷整理') # 有功数据来源
    output_dir = os.path.join(base_data_dir, '02_中间数据', '无功负荷整理')
    
    template_file = os.path.join(source_dir, '有功负荷处理模板.xlsx')
    DEFAULT_PF = 0.9 
    # 定义节点偏移量
    NODE_OFFSET = 1 
    # ===========================================

    # 1. 读取模板
    if not os.path.exists(template_file):
        # 尝试在源数据目录找
        alt_template = os.path.join(source_dir, '3.24有功整理.xlsx')
        if os.path.exists(alt_template): 
            template_file = alt_template
        else: 
            print(f"错误：找不到模板文件 [{template_file}]")
            return

    df_template = pd.read_excel(template_file, sheet_name=0)
    template_keys = clean_id_column(df_template['用户编号'])

    # 扫描文件 (从有功整理结果扫描)
    if not os.path.exists(p_source_dir):
        print(f"错误：找不到有功数据目录 [{p_source_dir}]")
        return

    files_p = glob.glob(os.path.join(p_source_dir, '*有功整理.xlsx'))
    prefixes = sorted(list(set([os.path.basename(f).replace('有功整理.xlsx','') for f in files_p])))
    
    if not prefixes:
        print(f"在 '{p_source_dir}' 目录下未找到 '*有功整理.xlsx' 文件。")
        return

    for date_prefix in prefixes:
        try:
            print(f"\n>>> 处理日期组: {date_prefix}")
            
            # --- 读取有功数据 (从中间数据) ---
            p_file = os.path.join(p_source_dir, f"{date_prefix}有功整理.xlsx")
            if not os.path.exists(p_file):
                print(f"  > 跳过：找不到有功数据文件 {p_file}")
                continue
            
            print(f"  > 读取有功数据: {p_file}")
            df_p = pd.read_excel(p_file)
            # 确保有 match_id 用于匹配 PF
            if '用户编号' in df_p.columns:
                df_p['match_id'] = clean_id_column(df_p['用户编号'])
            else:
                print(f"  > 警告：{p_file} 中缺少 '用户编号' 列，无法匹配功率因数。")
                continue
            
            # --- 读取 PF (保持原样，从源数据读取) ---
            dict_pf_pub, dict_pf_pri = {}, {}
            file_pf_pub = os.path.join(source_dir, f"{date_prefix}公.csv")
            if os.path.exists(file_pf_pub):
                t = pd.read_csv(file_pf_pub, encoding='gb18030')
                t['match_id'] = clean_id_column(t['台区编号'])
                t = t.drop_duplicates(subset=['match_id'], keep='first')
                dict_pf_pub = t.set_index('match_id').to_dict(orient='index')
            
            file_pf_pri = os.path.join(source_dir, f"{date_prefix}专.csv")
            if os.path.exists(file_pf_pri):
                t = pd.read_csv(file_pf_pri, encoding='gb18030')
                t['match_id'] = clean_id_column(t['用户编号'])
                t = t.drop_duplicates(subset=['match_id'], keep='first')
                dict_pf_pri = t.set_index('match_id').to_dict(orient='index')

            # --- 计算 Q ---
            # 直接在 df_p 的副本上修改数据
            df_organized = df_p.copy()
            
            # 遍历每一行计算 Q
            for idx, row in df_organized.iterrows():
                uid = row['match_id']
                
                # 查找 PF (优先查公有，再查专有)
                pf_row = None
                if uid in dict_pf_pub:
                    pf_row = dict_pf_pub[uid]
                elif uid in dict_pf_pri:
                    pf_row = dict_pf_pri[uid]
                
                for hour in range(24):
                    col = f"{hour}点"
                    if col not in row: continue
                    
                    p_val = row[col]
                    if pd.isna(p_val): p_val = 0
                    
                    # 获取该时刻的 PF
                    pf_val = DEFAULT_PF
                    if pf_row and col in pf_row and not pd.isna(pf_row[col]):
                        pf_val = pf_row[col]
                    
                    # 计算 Q
                    q_val = calculate_q(p_val, pf_val)
                    
                    # 更新 DataFrame
                    df_organized.at[idx, col] = q_val
            
            # 移除辅助列
            if 'match_id' in df_organized.columns:
                df_organized.drop(columns=['match_id'], inplace=True)
            
            # 保存明细 (明细表序号保持原样，方便核对)
            # 结果也保存到 '无功负荷整理' 文件夹
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            
            intermediate_file = os.path.join(output_dir, f"{date_prefix}无功整理.xlsx")
            df_organized.to_excel(intermediate_file, index=False)
            print(f"  > 成功生成中间数据: {intermediate_file}")

            # --- 调用去重脚本处理中间数据 ---
            # try:
            #     print(f"  > 调用 clean_hydro_duplicates 处理中间数据...")
            #     clean_hydro_duplicates.process_file(intermediate_file)
            #     # 重新读取处理后的数据
            #     df_organized = pd.read_excel(intermediate_file)
            # except Exception as e:
            #     print(f"  > 去重处理失败: {e}")

            # --- 生成无功负荷 (汇总表) ---
            # 1. 填充序号
            df_organized['序号'] = df_organized['序号'].ffill()
            df_agg = df_organized.dropna(subset=['序号']).copy() # Copy以免警告
            df_agg['序号'] = pd.to_numeric(df_agg['序号'], errors='coerce').astype(int)

            # ==========================================
            # 【核心修改】节点顺延逻辑
            # ==========================================
            print(f"  > 执行节点顺延: 所有序号 + {NODE_OFFSET}")
            df_agg['序号'] = df_agg['序号'] + NODE_OFFSET
            # ==========================================

            # 2. 聚合
            hour_cols = [f"{i}点" for i in range(24)]
            agg_dict = {c: 'sum' for c in hour_cols if c in df_agg.columns}
            if '数据日期' in df_agg.columns: agg_dict['数据日期'] = 'first'
            
            df_grouped = df_agg.groupby('序号', as_index=False).agg(agg_dict).sort_values('序号')
            
            df_grouped.to_excel(os.path.join(output_dir, f"{date_prefix}无功负荷.xlsx"), index=False)
            print(f"  > 成功生成 (Node 1 已预留): {date_prefix}无功负荷.xlsx")

        except Exception as e:
            print(f"  处理失败: {e}")

    print("\n所有无功处理任务完成！")

if __name__ == '__main__':
    main()