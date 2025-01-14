from tree_ import Tree
from tree import Tree 
# from graph_v1 import Graph, construct_graph
from graph_v2_lipo import Graph, construct_graph
from tools import *
import json

# 示例：从文件中读取数据
file_path = '/home/data1/lk/project/mol_generate/GDSS/data/qm9.csv'
SmilesList = read_smiles_from_file(file_path)

tree_construct_kenerl_method = ['edit_distance','graph']
threshold = 0.3

data_ = SmilesList


# 构建图
molecule_graph, atom_groups = construct_graph(data_, tree_construct_kenerl_method, threshold)
# 构建图


# 可视化图
# visualize_graph(molecule_graph, atom_groups, f'img/evolution_graph_{len(data_)}_{tree_construct_kenerl_method}_{threshold}_v2.png')

# Save the graph to a file
save_path = f'graph/evolution_graph_{len(data_)}_{tree_construct_kenerl_method}_{threshold}_v2.json'  # Specify the path where you want to save the graph
save_graph(molecule_graph, atom_groups, save_path)

# 输出图的边关系
# print(molecule_graph)
