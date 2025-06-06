import os

def delete_sense_files_in_folders(parent_path):
    try:
        # 获取父文件夹下的所有子文件夹
        subfolders = [f.path for f in os.scandir(parent_path) if f.is_dir()]

        for folder in subfolders:
            # 获取当前子文件夹下的所有文件
            files = os.listdir(folder)

            for file in files:
                file_path = os.path.join(folder, file)
                print("file_path:",file_path)
                # 检查文件是否以.sense扩展名结尾
                if file.endswith(".zip"):
                    # 删除文件
                    os.remove(file_path)
                    print(f"删除文件: {file_path}")

    except Exception as e:
        print(f"发生错误: {e}")

# 指定包含数字命名的文件夹的父文件夹路径
parent_folder = "/data/wb_project/scans/"

# 调用函数删除.sense文件
delete_sense_files_in_folders(parent_folder)