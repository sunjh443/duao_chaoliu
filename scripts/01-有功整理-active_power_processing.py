import pandas as pd
import os
import glob
import warnings

# 忽略读取Excel时的样式警告
warnings.filterwarnings("ignore")

def clean_id_column(series):
    """
    辅助函数：清洗ID列
    将各种格式（浮点、科学计数法、文本）统一转换为纯数字字符串，
    例如: 2394470.0 -> "2394470", "1763445" -> "1763445"
    """
    return pd.to_numeric(series, errors='coerce').fillna(0).astype(int).astype(str)

def get_date_prefixes(source_dir):
    """
    扫描指定目录下所有的 *公有.csv 和 *专有.csv 文件，
    提取出唯一的日期前缀 (例如 '7.16', '7.17')
    """
    files = glob.glob(os.path.join(source_dir, '*公有.csv')) + glob.glob(os.path.join(source_dir, '*专有.csv'))
    prefixes = set()
    for f in files:
        # 假设文件名格式为 "日期+类型.csv"，例如 "7.16公有.csv"
        name = os.path.basename(f)
        if '公有' in name:
            prefixes.add(name.replace('公有.csv', ''))
        elif '专有' in name:
            prefixes.add(name.replace('专有.csv', ''))
    return sorted(list(prefixes))

def main():
    # 切换工作目录到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # ================= 配置区域 =================
    # 脚本在 scripts/ 目录下，数据在 ../data/ 目录下
    base_data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    
    source_dir = os.path.join(base_data_dir, '01_源数据') # 源数据文件夹
    output_dir = os.path.join(base_data_dir, '02_中间数据', '有功负荷整理') # 结果输出文件夹
    template_file = os.path.join(source_dir, '有功负荷处理模板.xlsx')  # 模板文件名
    # ===========================================

    # 0. 准备输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 1. 读取模板 (仅读取 Sheet1)
    if not os.path.exists(template_file):
        print(f"错误：找不到模板文件 [{template_file}]")
        return

    print(f"正在读取模板: {template_file} ...")
    try:
        # sheet_name=0 默认读取第一个Sheet
        df_template = pd.read_excel(template_file, sheet_name=0)
    except Exception as e:
        print(f"模板读取失败: {e}")
        return
    
    # 预处理模板ID (作为匹配的基准 Key)
    template_keys = clean_id_column(df_template['用户编号'])
    
    # 2. 获取所有待处理的日期前缀
    prefixes = get_date_prefixes(source_dir)
    if not prefixes:
        print(f"在 '{source_dir}' 目录下未找到 '*公有.csv' 或 '*专有.csv' 文件。")
        return
    
    print(f"检测到 {len(prefixes)} 组待处理日期: {prefixes}")

    # 3. 循环处理每个日期组
    for date_prefix in prefixes:
        try:
            print(f"--------------------------------")
            print(f"正在处理日期组: {date_prefix} ...")
            
            # 准备存放合并后数据的列表
            df_list = []
            current_date_val = None
            
            # --- 读取公有文件 ---
            public_file = os.path.join(source_dir, f"{date_prefix}公有.csv")
            if os.path.exists(public_file):
                print(f"  > 读取: {public_file}")
                df_pub = pd.read_csv(public_file, encoding='gb18030')
                # 公有文件的ID在 '台区编号' 列
                if '台区编号' in df_pub.columns:
                    df_pub['match_id'] = clean_id_column(df_pub['台区编号'])
                    df_list.append(df_pub)
                    # 获取日期
                    if current_date_val is None and '数据日期' in df_pub.columns and not df_pub.empty:
                        current_date_val = df_pub['数据日期'].iloc[0]
            
            # --- 读取专有文件 ---
            private_file = os.path.join(source_dir, f"{date_prefix}专有.csv")
            if os.path.exists(private_file):
                print(f"  > 读取: {private_file}")
                df_pri = pd.read_csv(private_file, encoding='gb18030')
                # 专有文件的ID在 '用户编号' 列
                if '用户编号' in df_pri.columns:
                    df_pri['match_id'] = clean_id_column(df_pri['用户编号'])
                    df_list.append(df_pri)
                    # 获取日期 (如果公有文件没获取到)
                    if current_date_val is None and '数据日期' in df_pri.columns and not df_pri.empty:
                        current_date_val = df_pri['数据日期'].iloc[0]

            if not df_list:
                print(f"  > 跳过：未找到有效数据文件。")
                continue

            # --- 合并数据 ---
            # 将公有和专有数据垂直合并为一个大表
            df_combined = pd.concat(df_list, ignore_index=True)

            # --- 特殊过滤逻辑：优先保留“上网关口” ---
            # 对于存在“上网关口”的用户，只保留其“上网关口”的记录，丢弃该用户的其他记录
            ids_to_zero = [] # 记录需要置零的ID（因为模板中可能残留这些ID的数据）

            # 修改：优先使用“用户地址”作为判断依据，如果不存在则回退到“用户名称”
            filter_key = '用户地址' if '用户地址' in df_combined.columns else '用户名称'

            if '计量点用途类型' in df_combined.columns and filter_key in df_combined.columns:
                # 1. 找到所有拥有“上网关口”记录的关键字段（地址或名称）
                gateway_keys = df_combined[df_combined['计量点用途类型'] == '上网关口'][filter_key].unique()
                
                if len(gateway_keys) > 0:
                    print(f"  > 发现 {len(gateway_keys)} 个{filter_key}存在'上网关口'记录，将优先使用关口数据。")
                    
                    # 识别需要丢弃的行 (关键字段在 gateway_keys 中 且 类型不是 '上网关口')
                    mask_drop = (df_combined[filter_key].isin(gateway_keys)) & (df_combined['计量点用途类型'] != '上网关口')
                    
                    if mask_drop.any():
                        # 记录这些行的ID，以便后续在结果中置零
                        ids_to_zero = df_combined.loc[mask_drop, 'match_id'].unique().tolist()
                        print(f"  > 将清除 {len(ids_to_zero)} 个'售电侧结算'用户的残留数据。")
                        
                        # 执行过滤
                        df_combined = df_combined[~mask_drop]
                        print(f"  > 已过滤掉 {mask_drop.sum()} 行非关口重复数据。")
            
            # 创建 ID -> 数据 的映射 (去除重复ID，如果有重复以后面的为准)
            # 修改：增加 keep='first' 明确保留第一个出现的重复项
            source_map = df_combined.drop_duplicates(subset=['match_id'], keep='first').set_index('match_id')
            
            # --- 开始填充模板 ---
            df_result = df_template.copy()
            
            # 更新日期
            if current_date_val:
                df_result['数据日期'] = current_date_val
                print(f"  > 更新日期为: {current_date_val}")

            # 更新 0点 到 23点 的数据
            update_count = 0
            for hour in range(24):
                col_name = f"{hour}点"
                if col_name in source_map.columns:
                    mapped_values = template_keys.map(source_map[col_name])
                    # 仅更新匹配到的非空值
                    df_result[col_name] = mapped_values.fillna(df_result[col_name])

            # --- 清除被过滤掉的ID的数据 ---
            if ids_to_zero:
                # template_keys 是 df_result 中 '用户编号' 的清洗后版本
                # 找到 template_keys 中存在于 ids_to_zero 的行
                mask_zero = template_keys.isin(ids_to_zero)
                if mask_zero.any():
                     cols_to_zero = [f"{h}点" for h in range(24)]
                     cols_to_zero = [c for c in cols_to_zero if c in df_result.columns]
                     df_result.loc[mask_zero, cols_to_zero] = 0
                     print(f"  > 已置零 {mask_zero.sum()} 行被过滤的售电侧数据。")

            # --- 特殊处理：排除“杖锡水电站” ---
            # 找到“用户名称”列（根据实际列名修改 name_col）
            name_col = '用户名称' 
            target_name = '宁波市海曙杖锡水电站'
            
            if name_col in df_result.columns:
                # 找到匹配的行索引
                mask = df_result[name_col] == target_name
                if mask.any():
                    print(f"  > 发现特殊用户 [{target_name}]，正在将其功率置为 0 ...")
                    # 将 0点 到 23点 的数据全部置为 0
                    cols_to_zero = [f"{h}点" for h in range(24)]
                    # 确保这些列存在
                    cols_to_zero = [c for c in cols_to_zero if c in df_result.columns]
                    
                    df_result.loc[mask, cols_to_zero] = 0
                    print(f"  > 已排除 [{target_name}] 的功率数据。")
            else:
                print(f"  > 警告：模板中未找到列 [{name_col}]，无法排除 [{target_name}]。")

            # --- 保存结果 ---
            output_filename = os.path.join(output_dir, f"{date_prefix}有功整理.xlsx")
            df_result.to_excel(output_filename, index=False)
            print(f"  > 成功生成: {output_filename}")

        except Exception as e:
            print(f"  > 处理 {date_prefix} 组时失败: {e}")
            import traceback
            traceback.print_exc()

    print("--------------------------------")
    print("所有任务执行完毕。")

if __name__ == '__main__':
    main()