# COCO数据集的配置
MISSING_IDS = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]  # COCO数据集中缺失的类别ID
NUM_CLASSES_COCO = 80  # COCO数据集的类别数量

# 以下参数用于正确配置YOLOv3网络
# 原始YOLOv3网络有三个尺度，每个尺度有三个预定义的锚框
# 因此，总共需要九个锚框
# 按照惯例，锚框按从小到大排序
# 在COCO数据集中，总共有80个类别
# 因此属性总数为85，包括4个用于边界框，1个用于置信度
SCALES = 3  # 特征图尺度数量
NUM_ANCHORS_PER_SCALE = 3  # 每个尺度的锚框数量
ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]  # 预定义的锚框尺寸
assert len(ANCHORS) == SCALES * NUM_ANCHORS_PER_SCALE  # 确保锚框数量正确
NUM_CLASSES = NUM_CLASSES_COCO  # 类别数量
NUM_ATTRIB = 4 + 1 + NUM_CLASSES  # 每个预测框的属性数量：4(边界框)+1(置信度)+类别数
LAST_LAYER_DIM = NUM_ANCHORS_PER_SCALE * NUM_ATTRIB  # 最后一层的维度

# 训练参数
# IGNORE_THRESH是判断某个检测框是否被视为非目标的阈值
# 如YOLOv3论文所述："如果锚框不是最佳匹配但与真实目标的重叠度超过某个阈值，我们忽略该预测"
# 作者的真正意思是：如果原始检测框与所有真实框的最大IOU大于IGNORE_THRESH，
# 但不是与该真实框具有最佳IOU的候选检测框，则我们认为该检测框不会对损失函数产生贡献
# IGNORE_THRESH is the threshold whether to consider a certain detection is considered as non-object.
# As described in YOLOv3, "If the bounding box prior is not the best but does overlap a ground truth object
# by more than some threshold, we ignore the prediction." What the author really means is that:
# If the max IOU between the raw detection and all the ground truths is larger than IGNORE_THRESH, but not the best
# IOU among all the candidate detections with this ground truth, then we consider this detection will not contribute
# to the loss function.

IGNORE_THRESH = 0.5  # IOU忽略阈值

# NOOBJ_COEFF和COORD_COEFF是损失函数的超参数，如YOLOv1论文所述
# 这里我们使用以下两个值，它们能够产生与原始YOLOv3实现相当的结果
NOOBJ_COEFF = 0.2  # 非目标系数
COORD_COEFF = 5  # 坐标系数

# EPSILON用于避免计算中的不稳定性，如NaN或Inf
EPSILON = 1e-9  # 数值稳定性常数
