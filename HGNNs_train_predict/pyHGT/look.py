# import torch
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print(f"CUDA device name: {torch.cuda.get_device_name(device)}")
#     print(f"CUDA compute capability: {torch.cuda.get_device_capability(device)}")
# else:
#     print("CUDA is not available.")
    
# import torch
# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA version: {torch.version.cuda}")

# import torch_scatter
# print(torch_scatter.__version__)

# import torch
# if torch.cuda.is_available():
#     print(f"CUDA device count: {torch.cuda.device_count()}")
#     for i in range(torch.cuda.device_count()):
#         print(f"Device {i}: {torch.cuda.get_device_name(i)}")
#         print(f"Compute capability: {torch.cuda.get_device_capability(i)}")
# else:
#     print("CUDA is not available.")

# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())

import torch
from torch_geometric.data import HeteroData

data = torch.load('/user_data/yezy/zhangjm/pyHGT/PGMKG2.0/hetero_graph0725_standardized.pt', weights_only=False)
print('Data type:', type(data))

if isinstance(data, HeteroData):
    print('\n=== HeteroData Analysis ===')
    
    # 分析节点类型
    print('\nNode types:')
    node_types = data.node_types
    for node_type in node_types:
        print(f'  {node_type}:')
        if hasattr(data[node_type], 'x') and data[node_type].x is not None:
            print(f'    Features (x): {data[node_type].x.shape}')
        if hasattr(data[node_type], 'y') and data[node_type].y is not None:
            print(f'    Labels (y): {data[node_type].y.shape}')
        if hasattr(data[node_type], 'num_nodes'):
            print(f'    Number of nodes: {data[node_type].num_nodes}')
        # 检查其他属性
        for attr in dir(data[node_type]):
            if not attr.startswith('_') and attr not in ['x', 'y', 'num_nodes']:
                val = getattr(data[node_type], attr)
                if torch.is_tensor(val):
                    print(f'    {attr}: {val.shape}')
    
    # 分析边类型
    print('\nEdge types:')
    edge_types = data.edge_types
    for edge_type in edge_types:
        print(f'  {edge_type}:')
        edge_index = data[edge_type].edge_index
        print(f'    Edge index: {edge_index.shape}')
        print(f'    Number of edges: {edge_index.shape[1]}')
        if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
            print(f'    Edge attributes: {data[edge_type].edge_attr.shape}')
        # 检查其他边属性
        for attr in dir(data[edge_type]):
            if not attr.startswith('_') and attr not in ['edge_index', 'edge_attr']:
                val = getattr(data[edge_type], attr)
                if torch.is_tensor(val):
                    print(f'    {attr}: {val.shape}')
    
    # 元图信息
    print('\nMeta graph structure:')
    for edge_type in edge_types:
        src_type, relation, dst_type = edge_type
        print(f'  {src_type} --[{relation}]--> {dst_type}')

else:
    print('Data is not HeteroData, analyzing as dict...')
    for key, value in data.items():
        print(f'{key}: {type(value)}')
        if hasattr(value, 'shape'):
            print(f'  Shape: {value.shape}')
        elif isinstance(value, dict):
            print(f'  Dict keys: {list(value.keys())}')
            for k, v in value.items():
                if hasattr(v, 'shape'):
                    print(f'    {k}: {v.shape}')
                else:
                    print(f'    {k}: {type(v)}')