import pickle

data_path = "/home/hezhuolin/0_data/SSCBench/sscbench-waymo-pcd/Waymo-pcd/sscbench-waymo-pcd/stuff/seq_dict_000.pkl"

# 读取pkl文件
with open(data_path, 'rb') as file:
    data = pickle.load(file)

# 使用data进行操作
print(data)

import pdb; pdb.set_trace()
