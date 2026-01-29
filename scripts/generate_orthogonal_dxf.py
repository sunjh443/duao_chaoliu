import pandas as pd
import networkx as nx
import ezdxf
import math
try:
    import matplotlib.pyplot as plt
    from ezdxf.addons.drawing import RenderContext, Frontend
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
except ImportError:
    plt = None

def generate_orthogonal_dxf(input_file, output_file):
    # 1. 读取数据并构建图
    df = pd.read_csv(input_file)
    G = nx.Graph()
    
    for index, row in df.iterrows():
        start_id = row['起始序号']
        end_id = row['终点序号']
        G.add_edge(start_id, end_id)

    # 2. 拓扑分析：转换为有向树
    # 假设节点 1 为根节点
    root = 1
    if root not in G:
        # 如果没有节点1，找度数为1的节点作为备选根
        root = [n for n, d in G.degree() if d == 1][0]
    
    # 使用 BFS 构建树结构 (确定父子关系)
    tree = nx.bfs_tree(G, root)
    
    # 识别主干 (Main Feeder)
    # 计算所有节点到根的距离
    lengths = nx.single_source_shortest_path_length(tree, root)
    # 找到最远的节点
    farthest_node = max(lengths, key=lengths.get)
    # 回溯路径作为主干
    main_feeder_path = nx.shortest_path(tree, root, farthest_node)
    main_feeder_set = set(main_feeder_path)

    # 3. 计算正交布局坐标
    pos = {}
    # 根节点坐标
    pos[root] = (0, 0)
    
    # 辅助变量用于分配 Y 轴层级
    # 为了美观，我们可以交替在主干上下方分配分支
    # 或者简单地向上堆叠
    y_allocator = {'up': 100, 'down': -100}
    next_dir = 'up' # 轮询方向
    
    # 递归布局函数
    def layout_children(node, current_x, current_y, is_on_main_feeder):
        children = list(tree.successors(node))
        if not children:
            return

        # 优先处理主干上的子节点 (如果有)
        main_child = None
        for child in children:
            if child in main_feeder_set and is_on_main_feeder:
                main_child = child
                break
        
        # 布局主干子节点
        if main_child:
            pos[main_child] = (current_x + 100, current_y)
            layout_children(main_child, current_x + 100, current_y, True)
            
        # 布局分支子节点
        non_local_y_allocator = y_allocator # 引用外部变量
        non_local_next_dir = next_dir
        
        for child in children:
            if child == main_child:
                continue
            
            # 分支逻辑：
            # 分支的第一个节点，X 保持不变 (垂直引出)，Y 偏移
            # 为了防止重叠，我们需要分配一个新的 Y 层级
            # 这里简化处理：每个新的分支分配一个全新的 Y 高度
            # 更好的做法是检查当前 X 位置的 Y 占用情况，但全局递增最安全
            
            # 简单的 Y 分配策略：全局递增/递减
            # 这样保证不同分支绝对不会在 Y 轴重叠
            
            # 使用 nonlocal 关键字修改外部变量 (在 Python 3 中)
            # 这里用字典 hack
            direction = 'up' if y_allocator['up'] <= abs(y_allocator['down']) else 'down'
            
            if direction == 'up':
                branch_y = y_allocator['up']
                y_allocator['up'] += 100
            else:
                branch_y = y_allocator['down']
                y_allocator['down'] -= 100
                
            # 分支起点：(current_x, branch_y) -> 垂直线
            # 但为了美观，通常是先垂直走到 branch_y，再水平走？
            # 按照 Prompt：分支第一个子节点沿 Y 轴偏移。
            # 如果我们让第一个子节点坐标为 (current_x, branch_y)，那么连线就是垂直的。
            # 后续节点沿 X 轴延伸。
            
            pos[child] = (current_x, branch_y)
            
            # 递归处理该分支的后续节点
            # 注意：分支的后续节点不再视为 "Main Feeder"，它们沿 X 轴延伸
            # 传递 is_on_main_feeder=False
            # 这里的递归函数需要稍微调整逻辑：
            # 对于分支上的节点，它的子节点（延续）应该沿 X 轴走 (current_x + 100)
            # 它的分叉（二级分支）应该沿 Y 轴走
            
            # 我们定义一个内部递归，专门处理分支延伸
            layout_branch_extension(child, current_x, branch_y)

    def layout_branch_extension(node, x, y):
        children = list(tree.successors(node))
        if not children:
            return
            
        # 在分支上，我们需要选一个“主”延伸方向 (沿 X 轴)
        # 通常选最长的子树作为延伸
        # 简单起见，选第一个作为延伸，其他的作为二级分支
        
        # 排序子节点，选子树最深的作为延伸
        # 这里简单选第一个
        extension_child = children[0]
        
        # 延伸节点：沿 X 轴
        pos[extension_child] = (x + 100, y)
        layout_branch_extension(extension_child, x + 100, y)
        
        # 二级分支节点：沿 Y 轴再次偏移
        for child in children[1:]:
            # 分配新 Y
            direction = 'up' if y_allocator['up'] <= abs(y_allocator['down']) else 'down'
            if direction == 'up':
                sub_y = y_allocator['up']
                y_allocator['up'] += 100
            else:
                sub_y = y_allocator['down']
                y_allocator['down'] -= 100
            
            pos[child] = (x, sub_y) # 垂直引出
            layout_branch_extension(child, x, sub_y)

    # 开始布局
    layout_children(root, 0, 0, True)

    # 4. 生成 DXF
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    # 样式设置
    if 'SimSun' not in doc.styles:
        doc.styles.new('SimSun', dxfattribs={'font': 'simsun.ttc'})
    
    # 图层
    doc.layers.new(name='LINES', dxfattribs={'color': 7}) # 白色
    doc.layers.new(name='NODES', dxfattribs={'color': 1}) # 红色
    doc.layers.new(name='TEXT', dxfattribs={'color': 7}) # 黑色

    # 辅助绘图函数：L型连线
    def draw_L_connector(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        
        # 如果共线，直接画直线
        if x1 == x2 or y1 == y2:
            msp.add_line(p1, p2, dxfattribs={'layer': 'LINES'})
        else:
            # 直角连线：先 Y 后 X，或者先 X 后 Y
            # 根据我们的布局逻辑：
            # 主干 -> 分支：(x, 0) -> (x, y_new)。垂直线。
            # 分支 -> 延伸：(x, y) -> (x+100, y)。水平线。
            # 理论上我们的布局算法产生的父子连线应该都是正交的。
            # 只有一种情况：(x, y) -> (x, new_y) 是垂直的。
            # (x, y) -> (x+100, y) 是水平的。
            # 所以可能不需要 Polyline，但为了保险（防止浮点误差或逻辑漏洞），写一个
            
            # 策略：优先垂直走，再水平走 (或者反之)
            # 我们的分支逻辑是：从父节点垂直跳到新 Y。所以应该是 (x1, y1) -> (x1, y2) -> (x2, y2)
            # 但如果 x1 == x2 (垂直跳)，就是直线。
            # 检查一下：
            # 父 (x, y0), 子 (x, y1) -> 垂直线。
            # 父 (x, y1), 子 (x+100, y1) -> 水平线。
            # 似乎都是直线。
            
            # 无论如何，画个折线
            points = [(x1, y1), (x1, y2), (x2, y2)]
            msp.add_lwpolyline(points, dxfattribs={'layer': 'LINES'})

    # 绘制连线
    for u, v in tree.edges():
        if u in pos and v in pos:
            draw_L_connector(pos[u], pos[v])

    # 绘制节点和标签
    for node_id, (x, y) in pos.items():
        # 节点：实心圆 (使用 Hatch 或 简单圆环)
        # ezdxf 画实心圆比较麻烦，通常用 Donut (圆环) 也就是 LWPolyline with width
        # 或者简单的 Circle
        
        # 方法1: Circle
        msp.add_circle((x, y), radius=15, dxfattribs={'layer': 'NODES', 'color': 1}) # 半径大一点
        
        # 方法2: Solid Fill (Hatch) - 稍微复杂，先用圆代替，或者用 Solid
        # msp.add_point((x, y), dxfattribs={'layer': 'NODES'}) # 点太小
        
        # 标签：仅 ID
        label = str(int(node_id))
        msp.add_text(label, dxfattribs={
            'layer': 'TEXT',
            'style': 'SimSun',
            'height': 20, # 字体大小
        }).set_placement((x, y + 20), align=ezdxf.enums.TextEntityAlignment.CENTER) # 放在正上方

    doc.saveas(output_file)
    print(f"IEEE 33 风格拓扑图已生成: {output_file}")

    # 5. 生成 PDF
    if plt:
        try:
            pdf_file = output_file.replace('.dxf', '.pdf')
            print(f"正在生成 PDF: {pdf_file} ...")
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            ctx = RenderContext(doc)
            out = MatplotlibBackend(ax)
            Frontend(ctx, out).draw_layout(msp, finalize=True)
            fig.savefig(pdf_file)
            print(f"PDF 拓扑图已生成: {pdf_file}")
        except Exception as e:
            print(f"生成 PDF 失败: {e}")
    else:
        print("未安装 matplotlib，跳过 PDF 生成")

if __name__ == "__main__":
    input_csv = r"d:\projects\duaochaoliu\data\01_源数据\杜岙配电网_支路数据_含阻抗.csv"
    output_dxf = r"d:\projects\duaochaoliu\拓扑图_正交风格.dxf"
    generate_orthogonal_dxf(input_csv, output_dxf)
