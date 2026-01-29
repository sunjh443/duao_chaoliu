import pandas as pd
import ezdxf
import math
import os
import networkx as nx

def generate_dxf(input_file, output_file):
    # 加载数据
    df = pd.read_csv(input_file)
    
    # 创建 NetworkX 图对象
    G = nx.Graph()
    
    # 字典用于存储从 CSV 读取的初始坐标
    initial_pos = {} 
    
    # 第一遍扫描：收集边和已知坐标
    for index, row in df.iterrows():
        start_id = row['起始序号']
        end_id = row['终点序号']
        start_name = row['线路起始']
        end_name = row['线路终点']
        
        # 将节点和边添加到图中
        G.add_node(start_id, name=start_name)
        G.add_node(end_id, name=end_name)
        G.add_edge(start_id, end_id)
        
        # 坐标处理
        # 假设列索引: 5=起点_X, 6=起点_Y, 7=终点_X, 8=终点_Y
        try:
            sx = float(row.iloc[5])
            sy = float(row.iloc[6])
            ex = float(row.iloc[7])
            ey = float(row.iloc[8])
            
            if not math.isnan(sx) and not math.isnan(sy):
                initial_pos[start_id] = (sx, sy)
            if not math.isnan(ex) and not math.isnan(ey):
                initial_pos[end_id] = (ex, ey)
        except (ValueError, IndexError):
            pass

    # 处理主干节点 (1, 2, 3) 缺失坐标的情况
    # 逻辑：如果 1, 2, 3 缺失，则根据节点 4 沿 Y 轴正方向推演
    if 4 in initial_pos:
        x4, y4 = initial_pos[4]
        
        # 反向推演：3 在 4 上方，2 在 3 上方，1 在 2 上方
        # 按照要求使用 100 单位间距
        if 3 not in initial_pos:
            initial_pos[3] = (x4, y4 + 100)
        if 2 not in initial_pos:
            initial_pos[2] = (x4, y4 + 200)
        if 1 not in initial_pos:
            initial_pos[1] = (x4, y4 + 300)

    print(f"正在计算 {len(G.nodes)} 个节点的布局...")
    
    # 使用 Spring Layout (力导向算法) 计算布局
    # k: 节点间的最佳距离。增加到 100 以拉开间距。
    # scale: 坐标缩放因子。增加到 2000 以匹配较大的文字尺寸。
    # iterations: 模拟迭代次数。
    pos_layout = nx.spring_layout(G, pos=initial_pos, k=100, iterations=50, scale=2000, seed=42)

    # 创建 DXF 文档
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    # 设置文字样式
    if 'Songti' not in doc.styles:
        doc.styles.new('Songti', dxfattribs={'font': 'simsun.ttc'})
        
    # 图层设置
    doc.layers.new(name='LINES', dxfattribs={'color': 7}) # 白色/黑色
    doc.layers.new(name='NODES', dxfattribs={'color': 1}) # 红色
    doc.layers.new(name='TEXT', dxfattribs={'color': 3}) # 绿色
    
    # 绘制边 (连线)
    for u, v in G.edges():
        if u in pos_layout and v in pos_layout:
            start_pos = pos_layout[u]
            end_pos = pos_layout[v]
            msp.add_line(start_pos, end_pos, dxfattribs={'layer': 'LINES'})
            
    # 绘制节点和标签
    text_height = 15.0
    node_radius = 5.0
    text_offset = 8.0
    
    for node_id, pos in pos_layout.items():
        x, y = pos
        # 安全获取节点名称
        name = G.nodes[node_id].get('name', str(node_id))
        if pd.isna(name):
            name = str(node_id)
            
        # 绘制节点 (圆点)
        msp.add_circle((x, y), radius=node_radius, dxfattribs={'layer': 'NODES'})
        
        # 绘制标签
        # 格式: "ID: 名称"
        label = f"{int(node_id)}\n{name}"
        
        # 使用 MText 绘制多行文字
        msp.add_mtext(label, dxfattribs={
            'layer': 'TEXT',
            'style': 'Songti',
            'char_height': text_height,
            'insert': (x + text_offset, y + text_offset),
            'attachment_point': 7 # 左下角对齐 (Bottom Left)
        })
            
    doc.saveas(output_file)
    print(f"DXF 已生成: {output_file}")

if __name__ == "__main__":
    input_csv = r"d:\projects\duaochaoliu\data\01_源数据\杜岙配电网_支路数据_含阻抗.csv"
    output_dxf = r"d:\projects\duaochaoliu\拓扑图_generated.dxf"
    generate_dxf(input_csv, output_dxf)
