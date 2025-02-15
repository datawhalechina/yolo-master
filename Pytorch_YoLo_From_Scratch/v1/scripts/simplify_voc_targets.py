import os
import argparse
import xml.etree.ElementTree as Et

def parse_args():
    parser = argparse.ArgumentParser(description="简化 .xml 格式的目标标注文件，仅保留标签和边界框坐标。\n"
                                                 "对于每个图像中的边界框，会创建一个相同名称的 .csv 文件。\n"
                                                 "CSV 文件的第一行是表头，之后的行格式为 '<object>,<xmin>,<ymin>,<xmax>,<ymax>'。\n"
                                                 "旧的 .xml 标注文件将被删除。")

    parser.add_argument('dataset_path', type=str, help='PASCAL VOC 数据集的根目录（.tar 文件的下载位置）。')

    args = parser.parse_args()
    return args.dataset_path

def simplify_targets(dataset_path: str) -> None:
    """
    简化 .xml 目标标注文件，仅保留标签和边界框坐标，并转换为 .csv 文件。转换完成后删除 .xml 文件。
    训练集和测试集中均忽略困难目标。

    :param dataset_path: PASCAL VOC 数据集的根目录（.tar 文件的下载位置）。
    """
    for dataset_part_dir in ['train', 'test']:
        annot_dir_name = os.path.join(dataset_path, "VOC_Detection", dataset_part_dir, "targets")
        for annot_file_name in os.listdir(annot_dir_name):

            csv_filename = os.path.join(annot_dir_name, f'{annot_file_name[:-4]}.csv')
            xml_filename = os.path.join(annot_dir_name, annot_file_name)

            with open(csv_filename, 'w') as csv_file:
                csv_file.write("object,xmin,ymin,xmax,ymax")
                with open(xml_filename, 'r') as xml_file:
                    for obj in Et.parse(xml_file).getroot().findall('object'):
                        difficult_flag = obj.find('difficult').text
                        if difficult_flag == '0':
                            label = obj.find('name').text
                            bbox_dict = {bndbox.tag: bndbox.text for bndbox in obj.find('bndbox')}
                            csv_file.write(f"\n{label},"
                                           f"{bbox_dict['xmin']},{bbox_dict['ymin']},"
                                           f"{bbox_dict['xmax']},{bbox_dict['ymax']}")
            os.remove(xml_filename)

if __name__ == '__main__':
    dataset_path = parse_args()
    simplify_targets(dataset_path)
