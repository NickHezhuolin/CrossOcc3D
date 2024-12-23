import os

# 源路径和目标路径
source_path_2d = "/home/hezhuolin/0_data/SSCbench/sscbench-kitti/kitti360/sscbench-kitti/data_2d_raw/"
source_path_3d = "/home/hezhuolin/0_data/SSCbench/sscbench-kitti/kitti360/sscbench-kitti/data_3d_raw/"
target_path = "/home/hezhuolin/1_work_dir/CrossOcc3D/data/SSCBenchKITTI360/data_2d_raw/"

# 要处理的文件夹名称
folders = [
    "2013_05_28_drive_0000_sync",
    "2013_05_28_drive_0002_sync",
    "2013_05_28_drive_0003_sync",
    "2013_05_28_drive_0004_sync",
    "2013_05_28_drive_0005_sync",
    "2013_05_28_drive_0006_sync",
    "2013_05_28_drive_0007_sync",
    "2013_05_28_drive_0009_sync",
    "2013_05_28_drive_0010_sync"
]

# 在目标路径下创建文件夹
for folder in folders:
    folder_path = os.path.join(target_path, folder)
    os.makedirs(folder_path, exist_ok=True)
    print(f"创建文件夹: {folder_path}")

def create_subfolder_symlinks(source_folder, target_folder):
    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)

    # 遍历源文件夹中的所有项目
    for item in os.listdir(source_folder):
        source_item = os.path.join(source_folder, item)
        target_item = os.path.join(target_folder, item)

        # 只为子文件夹创建软链接
        if os.path.isdir(source_item):
            if not os.path.exists(target_item):
                os.symlink(source_item, target_item)
                print(f"创建子文件夹软链接: {source_item} -> {target_item}")
            else:
                print(f"子文件夹软链接已存在，跳过: {target_item}")

# 为每个主文件夹创建软链接
for folder in folders:
    source_folder = os.path.join(source_path_2d, folder)
    target_folder = os.path.join(target_path, folder)
    
    if os.path.exists(source_folder):
        create_subfolder_symlinks(source_folder, target_folder)
    else:
        print(f"源文件夹不存在: {source_folder}")

print("所有子文件夹软链接创建完成。")

# 创建符号链接
for folder in folders:
    source_2d_folder = os.path.join(source_path_2d, folder)
    source_3d_folder = os.path.join(source_path_3d, folder)
    target_folder = os.path.join(target_path, folder)

    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)

    # 获取源文件夹中的文件列表
    if os.path.exists(source_2d_folder):
        for file_name in os.listdir(source_2d_folder):
            source_file = os.path.join(source_2d_folder, file_name)
            target_file = os.path.join(target_folder, file_name)
            if not os.path.exists(target_file):
                os.symlink(source_file, target_file)
                print(f"Created symlink for 2D: {source_file} -> {target_file}")

    if os.path.exists(source_3d_folder):
        for file_name in os.listdir(source_3d_folder):
            source_file = os.path.join(source_3d_folder, file_name)
            target_file = os.path.join(target_folder, file_name)
            if not os.path.exists(target_file):
                os.symlink(source_file, target_file)
                print(f"Created symlink for 3D: {source_file} -> {target_file}")

print("All symlinks created successfully.")