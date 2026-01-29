import pandas as pd
import os
import glob

def process_file(file_path):
    print(f"--------------------------------------------------")
    print(f"Processing file: {file_path}")

    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Identify hour columns
    hour_cols = [f"{i}点" for i in range(24)]

    # Check if columns exist
    missing_cols = [col for col in hour_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        return

    # 判断文件类型（有功 vs 无功）
    file_name = os.path.basename(file_path)
    is_reactive_file = "无功" in file_name

    if is_reactive_file:
        print("检测到无功文件，使用替代方法识别水电站...")
        # 对于无功文件，查找对应的有功文件来识别水电站
        # 需要替换目录名和文件名
        active_file_path = file_path.replace("无功负荷整理", "有功负荷整理").replace("无功整理", "有功整理")

        if os.path.exists(active_file_path):
            print(f"读取对应有功文件: {active_file_path}")
            try:
                df_active = pd.read_excel(active_file_path)
                # 使用有功文件的负值来识别水电站行
                is_hydro = df_active[hour_cols].min(axis=1) < -0.001
                print(f"从有功文件识别到 {is_hydro.sum()} 个水电站节点")
            except Exception as e:
                print(f"读取有功文件失败: {e}，跳过去重处理")
                return
        else:
            print(f"未找到对应有功文件: {active_file_path}，跳过去重处理")
            return
    else:
        # 对于有功文件，使用负值识别水电站
        # Identify hydro nodes (rows with negative values in any hour column)
        # Using a threshold of -0.001 to avoid floating point issues with 0
        is_hydro = df[hour_cols].min(axis=1) < -0.001

    hydro_df = df[is_hydro].copy()
    non_hydro_df = df[~is_hydro]

    print(f"Total rows: {len(df)}")
    print(f"Hydro nodes found: {len(hydro_df)}")

    if len(hydro_df) == 0:
        print("No hydro nodes found.")
        return

    # Find duplicates based on hour columns values
    # subset=hour_cols considers only the power values for duplication check
    # keep='first' marks duplicates as True except for the first occurrence
    duplicates_mask = hydro_df.duplicated(subset=hour_cols, keep='first')

    duplicates_count = duplicates_mask.sum()
    print(f"Duplicate hydro entries found (same values): {duplicates_count}")
    
    if duplicates_count > 0:
        # Get the IDs of duplicates for reporting
        duplicate_rows = hydro_df[duplicates_mask]
        print("Removing the following duplicate IDs (keeping the first occurrence):")
        if '序号' in df.columns:
            print(duplicate_rows['序号'].tolist())
        else:
            print(duplicate_rows.index.tolist())

        # Keep only non-duplicates from hydro_df
        hydro_df_cleaned = hydro_df[~duplicates_mask]
        
        # Combine back with non-hydro nodes
        df_cleaned = pd.concat([non_hydro_df, hydro_df_cleaned])
        
        if '序号' in df_cleaned.columns:
            df_cleaned = df_cleaned.sort_values('序号')
        
        print(f"Rows after cleaning: {len(df_cleaned)}")
        
        # Save back to file
        print(f"Saving cleaned data to: {file_path}")
        df_cleaned.to_excel(file_path, index=False)
        print("Done.")
        
    else:
        print("No duplicates found to remove.")

def main():
    # Configurations to process
    configs = [
        {
            'dir': r'D:\projects\duaochaoliu\data\02_中间数据\有功负荷整理',
            'pattern': '*有功整理.xlsx'
        },
        {
            'dir': r'D:\projects\duaochaoliu\data\02_中间数据\无功负荷整理',
            'pattern': '*无功整理.xlsx'
        }
    ]
    
    for config in configs:
        data_dir = config['dir']
        file_pattern = config['pattern']
        
        print(f"\n==================================================")
        print(f"Processing directory: {data_dir}")
        print(f"Pattern: {file_pattern}")
        
        if not os.path.exists(data_dir):
            print(f"Directory not found: {data_dir}")
            continue

        # Find all matching files
        pattern_path = os.path.join(data_dir, file_pattern)
        files = glob.glob(pattern_path)
        
        if not files:
            print(f"No files found matching pattern: {pattern_path}")
            continue
            
        print(f"Found {len(files)} files to process.")
        
        for file_path in files:
            process_file(file_path)

if __name__ == "__main__":
    main()
