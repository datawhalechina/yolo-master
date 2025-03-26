import os
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(
        description="简化 PASCAL VOC 标注文件",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('dataset_path', type=str, 
                      help='PASCAL VOC 数据集根目录\n'
                           '示例结构：\n'
                           'VOC_Detection/\n'
                           '├── train/\n'
                           '│   ├── images/\n'
                           '│   └── targets/\n'
                           '└── test/\n'
                           '    ├── images/\n'
                           '    └── targets/')
    return parser.parse_args()

def print_summary(stats):
    print("\n转换摘要：")
    print(f"总处理数据集: {len(stats)} 个（train/test）")
    for dataset, data in stats.items():
        print(f"\n{dataset.upper()} 数据集:")
        print(f"  - 转换文件数: {data['total_files']}")
        print(f"  - 成功转换: {data['converted_files']}")
        print(f"  - 失败文件: {data['failed_files']}")
        print(f"  - 有效目标数: {data['valid_objects']}")
        print(f"  - 跳过困难目标: {data['difficult_skipped']}")
        print(f"  - 删除旧文件: {data['deleted_xml']} 个")

def simplify_targets(dataset_path: str) -> None:
    stats = defaultdict(lambda: {
        'total_files': 0,
        'converted_files': 0,
        'failed_files': 0,
        'valid_objects': 0,
        'difficult_skipped': 0,
        'deleted_xml': 0
    })

    for dataset_part in ['train', 'test']:
        print(f"\n{'='*30}\n开始处理 {dataset_part.upper()} 数据集\n{'='*30}")
        
        annot_dir = os.path.join(dataset_path, "VOC_Detection", dataset_part, "targets")
        if not os.path.exists(annot_dir):
            print(f"! 目录不存在: {annot_dir}")
            continue

        xml_files = [f for f in os.listdir(annot_dir) if f.endswith('.xml')]
        stats[dataset_part]['total_files'] = len(xml_files)
        
        if not xml_files:
            print(f"! 未找到 XML 文件: {annot_dir}")
            continue

        print(f"找到 {len(xml_files)} 个 XML 文件，开始转换...")

        for idx, xml_file in enumerate(xml_files, 1):
            xml_path = os.path.join(annot_dir, xml_file)
            csv_file = f"{xml_path[:-4]}.csv"
            current_stat = stats[dataset_part]

            try:
                print(f"\n[{idx}/{len(xml_files)}] 处理文件: {xml_file}")
                
                # 解析 XML
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                valid_objs = 0
                difficult_objs = 0

                # 写入 CSV
                with open(csv_file, 'w', encoding='utf-8') as f:
                    f.write("object,xmin,ymin,xmax,ymax\n")
                    for obj in root.findall('object'):
                        if obj.find('difficult').text == '1':
                            difficult_objs += 1
                            continue
                            
                        label = obj.find('name').text
                        bbox = obj.find('bndbox')
                        f.write(f"{label},"
                                f"{bbox.find('xmin').text},"
                                f"{bbox.find('ymin').text},"
                                f"{bbox.find('xmax').text},"
                                f"{bbox.find('ymax').text}\n")
                        valid_objs += 1

                # 更新统计
                current_stat['converted_files'] += 1
                current_stat['valid_objects'] += valid_objs
                current_stat['difficult_skipped'] += difficult_objs
                
                # 删除旧文件
                os.remove(xml_path)
                current_stat['deleted_xml'] += 1
                
                print(f"转换成功 | 有效目标: {valid_objs} | 跳过困难: {difficult_objs}")

            except Exception as e:
                print(f"! 处理失败: {xml_file}\n错误信息: {str(e)}")
                current_stat['failed_files'] += 1
                if os.path.exists(csv_file):
                    os.remove(csv_file)

    print_summary(stats)

if __name__ == '__main__':
    args = parse_args()
    
    if not os.path.exists(os.path.join(args.dataset_path, "VOC_Detection")):
        print(f"错误：数据集目录结构不正确，请确认路径包含 VOC_Detection 目录")
        sys.exit(1)
        
    print("="*50)
    print("开始转换 PASCAL VOC 标注文件")
    print(f"数据集路径: {os.path.abspath(args.dataset_path)}")
    print("="*50)
    
    simplify_targets(args.dataset_path)
    print("\n所有处理完成！")