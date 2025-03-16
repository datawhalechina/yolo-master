import sys
import os
import subprocess
import tarfile
import argparse

def main():
    parser = argparse.ArgumentParser(description="下载PASCAL VOC数据集")
    parser.add_argument("dir_path", help="存储数据集的目录路径")
    args = parser.parse_args()

    os.makedirs(args.dir_path, exist_ok=True)

    # 安装 git LFS
    try:
        subprocess.run(['git', 'lfs', 'install'], 
                      check=True, 
                      stderr=subprocess.DEVNULL,
                      stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("[INFO] Git LFS 可能已经安装")

    # 构建认证URL
    repo_url = 'https://www.modelscope.cn/datasets/yolo_master/VOC0712.git'

    # 克隆仓库
    clone_result = subprocess.run(
        ['git', 'clone', repo_url, args.dir_path],
        capture_output=True,
        text=True
    )

    if clone_result.returncode != 0:
        if "already exists" in clone_result.stderr:
            print("[WARNING] 目标目录已存在，继续后续操作")
        else:
            print(f"[ERROR] 仓库克隆失败: {clone_result.stderr}")
            sys.exit(1)

    # 验证必须的压缩文件
    required_tars = [
        'VOCtrainval_11-May-2012.tar',
        'VOCtrainval_06-Nov-2007.tar',
        'VOCtest_06-Nov-2007.tar'
    ]
    
    missing_files = []
    for f in required_tars:
        if not os.path.exists(os.path.join(args.dir_path, f)):
            missing_files.append(f)
    
    if missing_files:
        print(f"[ERROR] 缺少必要文件: {', '.join(missing_files)}")
        sys.exit(1)

    # 解压并清理
    for tar_name in required_tars:
        tar_path = os.path.join(args.dir_path, tar_name)
        print(f"[PROGRESS] 正在解压 {tar_name}...")
        
        try:
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=args.dir_path)
            os.remove(tar_path)
        except Exception as e:
            print(f"[ERROR] 解压失败: {str(e)}")
            sys.exit(1)

    print("[SUCCESS] 数据集下载解压完成")
    print(f"最终数据集路径: {os.path.abspath(args.dir_path)}")

if __name__ == "__main__":
    main()