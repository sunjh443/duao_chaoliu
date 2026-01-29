import pandas as pd
import os
import glob
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

def main():
    # 路径强行定位
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    base_data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    input_folder = os.path.join(base_data_dir, '02_中间数据', '有功负荷整理')
    output_folder = os.path.join(base_data_dir, '02_中间数据', '负荷整理')
    
    input_pattern = os.path.join(input_folder, '*有功整理.xlsx')

    # ================= 配置区域 =================
    # 定义节点偏移量 (Node 1 留给发电机，所有负荷顺延 1 位)
    NODE_OFFSET = 1 
    # ===========================================

    files = glob.glob(input_pattern)
    if not files:
        print(f"未找到 [{input_pattern}] 文件。")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"找到 {len(files)} 个文件，开始处理 (偏移量 +{NODE_OFFSET})...")

    hour_cols = [f"{i}点" for i in range(24)]

    for file_path in files:
        try:
            print(f"--------------------------------")
            print(f"正在读取: {file_path}")
            df = pd.read_excel(file_path)

            if '序号' not in df.columns:
                print(f"  > 跳过：无 [序号] 列")
                continue

            # 1. 填充与清洗序号
            df['序号'] = df['序号'].ffill()
            df = df.dropna(subset=['序号'])
            df['序号'] = pd.to_numeric(df['序号'], errors='coerce').astype(int)

            # ==========================================
            # 【核心修改】节点顺延逻辑
            # ==========================================
            print(f"  > 执行节点顺延: 所有序号 + {NODE_OFFSET}")
            df['序号'] = df['序号'] + NODE_OFFSET
            # ==========================================

            cols_to_sum = [c for c in hour_cols if c in df.columns]
            if not cols_to_sum: continue

            agg_dict = {col: 'sum' for col in cols_to_sum}
            if '数据日期' in df.columns: agg_dict['数据日期'] = 'first'

            # 分组聚合
            df_grouped = df.groupby('序号', as_index=False).agg(agg_dict)
            df_grouped = df_grouped.sort_values('序号')

            # 保存
            base_name = os.path.basename(file_path)
            output_filename = base_name.replace('有功整理', '负荷整理')
            output_path = os.path.join(output_folder, output_filename)

            df_grouped.to_excel(output_path, index=False)
            print(f"  > 成功生成 (Node 1 已预留): {output_path}")

        except Exception as e:
            print(f"  > 失败: {e}")

    print("--------------------------------")
    print("有功负荷累加完成。")

if __name__ == '__main__':
    main()