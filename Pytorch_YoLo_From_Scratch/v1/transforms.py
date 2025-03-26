import torch as th
from torch.nn.functional import one_hot
from torchvision import transforms
import torchvision.transforms.functional as fT
import PIL.Image as Image
from typing import Tuple, Optional, List


class Resize:
    """
    可调用的图像缩放类，在调用时会调整图像大小并相应缩放边界框坐标。
    """

    def __init__(self, output_size: int) -> None:
        """
        初始化变换后的图像维度d。图像变换后将具有(d x d)形状。

        :param output_size: 变换后的图像维度。
        """
        self.d = output_size

    def __call__(self, sample: Tuple[Image.Image, th.Tensor]
                 ) -> Tuple[Image.Image, List[Tuple[float, float]], th.Tensor]:
        """
        将图像调整为(d x d)形状并转换边界框坐标。
        在包含N个对象的图像中，目标张量形状为(N x 5)，每个对象的格式为：
        <分类ID>, <x_min>, <y_min>, <x_max>, <y_max>。对于(h x w)图像，坐标更新方式：

        | x' = x * d / w
        | y' = y * d / h

        :param sample: 包含图像及其对应目标的元组
        :return: 调整后的图像、包含所有有效像素的掩码区域([0,d]范围)、缩放后的坐标
        """
        img, target = sample
        w, h = img.size

        img = fT.resize(img, (self.d, self.d))
        target[:, [1, 3]] *= self.d / w
        target[:, [2, 4]] *= self.d / h

        mask = [(0, self.d), (0, self.d)]
        return img, mask, target


class RandomScaleTranslate:
    """
    可调用随机缩放平移类，用于调整图像大小并转换边界框坐标。为增强数据集，对每张图像随机选择以下操作：
    - 直接缩放
    - 缩小后缩放
    - 放大后缩放

    当缩小图像时，会用零填充。为避免扭曲这些零值（如颜色抖动、归一化），返回掩码标识有效区域。
    """
    def __init__(self,
                 output_size: int,
                 jitter: float,
                 resize_p: float,
                 zoom_out_p: float,
                 zoom_in_p: float) -> None:
        """
        初始化变换后的图像维度d，存储抖动因子用于随机缩放平移，使用概率参数选择操作。

        :param output_size: 变换后的图像维度
        :param jitter: 控制随机缩放和平移的因子
        :param resize_p: 直接缩放操作的概率
        :param zoom_out_p: 缩小后缩放操作的概率
        :param zoom_in_p: 放大后缩放操作的概率
        """
        self.d = output_size
        self.jitter = jitter
        self.t_probs = th.cumsum(th.Tensor([resize_p, zoom_out_p, zoom_in_p]), dim=0)

    def __call__(self, sample: Tuple[Image.Image, th.Tensor]
                 ) -> Tuple[Image.Image, List[Tuple[float, float]], th.Tensor]:
        """
        从均匀分布中采样决定执行哪个操作，调整图像尺寸并转换边界框坐标，返回掩码标识有效区域。
        对于过小的边界框会进行移除。

        :param sample: 包含图像及其目标的元组
        :return: 变换后的图像、掩码区域及更新后的目标
        """

        transform_prob = th.rand(1)
        if transform_prob < self.t_probs[0]:                    # 直接缩放
            img, mask, target = self._resize(sample)
        elif transform_prob < self.t_probs[1]:                  # 缩小后缩放
            img, mask, target = self._zoom_out(sample)
        else:                                                   # 放大后缩放
            img, mask, target = self._zoom_in(sample)

        # 移除过小的边界框
        bboxes_w = target[:, 3] - target[:, 1]
        bboxes_h = target[:, 4] - target[:, 2]
        threshold = 0.001 * self.d
        valid_bboxes = th.logical_not(th.logical_or(bboxes_w < threshold, bboxes_h < threshold))
        target = target[valid_bboxes]
        return img, mask, target

    def _resize(self, sample: Tuple[Image.Image, th.Tensor]
                ) -> Tuple[Image.Image, List[Tuple[float, float]], th.Tensor]:
        """
        执行与Resize类相同的直接缩放逻辑。

        :param sample: 包含图像及其目标的元组
        :return: 缩放后的图像、全尺寸掩码及更新后的目标
        """
        img, target = sample
        w, h = img.size

        img = fT.resize(img, (self.d, self.d))
        target[:, [1, 3]] *= self.d / w
        target[:, [2, 4]] *= self.d / h

        mask = [(0, self.d), (0, self.d)]
        return img, mask, target

    def _zoom_out(self, sample: Tuple[Image.Image, th.Tensor]
                  ) -> Tuple[Image.Image, List[Tuple[float, float]], th.Tensor]:
        """
        通过随机采样新宽高比实现缩小操作：
        1. 计算新宽高比new_ar = rand_w / rand_h
        2. 根据宽高比确定新尺寸(d, k)或(k, d)
        3. 随机平移并填充图像
        4. 转换坐标并生成掩码标识有效区域

        :param sample: 包含图像及其目标的元组
        :return: 缩小后的图像、掩码区域及更新后的目标
        """

        img, target = sample
        w, h = img.size

        dw = w * self.jitter
        dh = h * self.jitter
        rand_w = w + th.Tensor(1).uniform_(-dw, dw)
        rand_h = h + th.Tensor(1).uniform_(-dh, dh)
        new_ar = rand_w / rand_h

        if new_ar < 1:
            nh = self.d
            nw = int(nh * new_ar + 0.5)
        else:
            nw = self.d
            nh = int(nw / new_ar + 0.5)

        dx = th.randint(low=0, high=self.d - nw + 1, size=(1,)).item()
        dy = th.randint(low=0, high=self.d - nh + 1, size=(1,)).item()

        img = fT.resize(img, (nh, nw))
        target[:, [1, 3]] *= nw / w
        target[:, [2, 4]] *= nh / h

        img = fT.pad(img, padding=[dx, dy, self.d - nw - dx, self.d - nh - dy])
        target[:, [1, 3]] += dx
        target[:, [2, 4]] += dy

        mask = [(dx, dx + nw), (dy, dy + nh)]
        return img, mask, target

    def _zoom_in(self, sample: Tuple[Image.Image, th.Tensor]
                 ) -> Tuple[Image.Image, List[Tuple[float, float]], th.Tensor]:
        """
        随机采样裁剪区域并放大：
        1. 在原图上随机裁剪子区域
        2. 将子区域缩放到目标尺寸
        3. 移除不可见的边界框，裁剪部分可见的框进行坐标修正

        :param sample: 包含图像及其目标的元组
        :return: 放大后的图像、全尺寸掩码及更新后的目标
        """
        img, target = sample
        w, h = img.size

        nw = int(th.Tensor(1).uniform_((1 - self.jitter) * w, w) + 0.5)
        nh = int(th.Tensor(1).uniform_((1 - self.jitter) * h, h) + 0.5)
        dx = int(th.Tensor(1).uniform_(0, w - nw + 1) + 0.5)
        dy = int(th.Tensor(1).uniform_(0, h - nh + 1) + 0.5)

        img = fT.resized_crop(img, top=dy, left=dx, height=nh, width=nw, size=(self.d, self.d))

        target[:, [1, 3]] -= dx
        target[:, [2, 4]] -= dy
        target[:, [1, 3]] *= self.d / nw
        target[:, [2, 4]] *= self.d / nh

        # 移除完全不可见的边界框
        target = target[th.logical_not(th.logical_or(th.logical_or(target[:, 3] < 0, target[:, 1] > self.d),
                                                     th.logical_or(target[:, 4] < 0, target[:, 2] > self.d)))]

        # 修正部分可见框的坐标
        target[:, [1, 2]] = target[:, [1, 2]].clamp(min=0)
        target[:, [3, 4]] = target[:, [3, 4]].clamp(max=self.d)

        mask = [(0, self.d), (0, self.d)]
        return img, mask, target


class RandomColorJitter:
    """
    可调用随机颜色抖动类，对输入图像进行颜色扭曲，目标值保持不变。
    """

    def __init__(self, hue: float, sat: float, exp: float):
        """
        初始化色调、饱和度和曝光参数。

        :param hue: 色调参数，从[-hue, hue]均匀采样
        :param sat: 饱和度参数，从[1/sat, sat]均匀采样
        :param exp: 曝光参数，从[1/exp, exp]均匀采样
        """
        self.hue = hue
        self.sat = sat
        self.exp = exp

    def __call__(self, sample: Tuple[Image.Image, List[Tuple[float, float]], th.Tensor]
                 ) -> Tuple[Image.Image, List[Tuple[float, float]], th.Tensor]:
        """
        在HSV空间调整颜色：
        - 色调：加减随机值并循环处理
        - 饱和度：乘以随机因子并截断
        - 曝光：乘以随机因子并截断

        :param sample: 包含图像、掩码及目标的元组
        :return: 颜色调整后的图像及未变化的目标
        """
        rand_hue = th.Tensor(1).uniform_(-self.hue, self.hue)
        rand_sat = th.Tensor(1).uniform_(1 / self.sat, self.sat)
        rand_exp = th.Tensor(1).uniform_(1 / self.exp, self.exp)

        rgb_img, mask, target = sample
        hsv_img = rgb_img.convert('HSV')
        hsv_tensor = fT.to_tensor(hsv_img)

        mask_x, mask_y = mask
        masked_hsv_tensor = hsv_tensor[:, mask_y[0]:mask_y[1], mask_x[0]:mask_x[1]]

        # 调整色调
        masked_hsv_tensor[0, :, :] += rand_hue
        masked_hsv_tensor[0, :, :] += (1. * (masked_hsv_tensor[0, :, :] < 0) - 1. * \
                                       (masked_hsv_tensor[0, :, :] > 1)) * th.ones_like(masked_hsv_tensor[0, :, :])
        # 调整饱和度
        masked_hsv_tensor[1, :, :] *= rand_sat
        masked_hsv_tensor[1, :, :] = masked_hsv_tensor[1, :, :].clamp(max=1.0)

        # 调整曝光
        masked_hsv_tensor[2, :, :] *= rand_exp
        masked_hsv_tensor[2, :, :] = masked_hsv_tensor[2, :, :].clamp(max=1.0)

        hsv_img = fT.to_pil_image(hsv_tensor, mode='HSV')
        rgb_img = hsv_img.convert('RGB')

        return rgb_img, mask, target


class RandomHorizontalFlip:
    """
    可调用随机水平翻转类，随机决定是否翻转图像，同时调整边界框坐标和掩码。
    """
    def __init__(self, p: float) -> None:
        """
        初始化翻转概率。

        :param p: 应用水平翻转的概率
        """
        self.p = p

    def __call__(self, sample: Tuple[Image.Image, List[Tuple[float, float]], th.Tensor]
                 ) -> Tuple[Image.Image, List[Tuple[float, float]], th.Tensor]:
        """
        按概率p翻转图像，调整x坐标并更新掩码。

        :param sample: 包含图像、掩码及目标的元组
        :return: 翻转后的图像或原始样本
        """
        apply_transform = th.rand(1) < self.p
        if not apply_transform:
            return sample

        img, mask, target = sample
        w = img.size[0]

        target[:, [1, 3]] = w - target[:, [3, 1]]
        img = fT.hflip(img)

        start_x, end_x = mask[0]
        mask[0] = (w - end_x, w - start_x)

        return img, mask, target


class ToYOLOTensor:
    """
    可调用YOLO张量转换类，将目标转换为YOLO格式，图像转为张量，可选归一化。
    """

    def __init__(self, S: int, C: int, normalize: Optional[List] = None) -> None:
        """
        初始化YOLO网格参数。

        :param S: 网格尺寸（S x S）
        :param C: 数据集类别数
        :param normalize: 各通道的均值和标准差
        """
        self.S = S
        self.C = C
        self.normalize = normalize

    def __call__(self, sample: Tuple[Image.Image, List[Tuple[float, float]], th.Tensor]
                 ) -> Tuple[th.Tensor, th.Tensor]:
        """
        转换目标为(SxS)网格格式：
        - 每个单元格包含存在概率、one-hot类别、归一化中心坐标和尺寸

        :param sample: 包含图像、掩码及目标的元组
        :return: 图像张量和YOLO格式目标
        """
        img, mask, target = sample
        w, h = img.size

        cell_w = w / self.S
        cell_h = h / self.S

        center_x = (target[:, 1] + target[:, 3]) / 2
        center_y = (target[:, 2] + target[:, 4]) / 2
        bndbox_w = target[:, 3] - target[:, 1]
        bndbox_h = target[:, 4] - target[:, 2]

        label = target[:, 0].long()
        center_col = th.div(center_x, cell_w, rounding_mode="trunc").long()
        center_row = th.div(center_y, cell_h, rounding_mode="trunc").long()
        norm_center_x = (center_x % cell_w) / cell_w
        norm_center_y = (center_y % cell_h) / cell_h
        norm_bndbox_w = bndbox_w / w
        norm_bndbox_h = bndbox_h / h

        target = th.zeros((self.S, self.S, self.C + 5))
        target[center_row, center_col, :] = th.cat([th.ones((label.shape[0], 1)),
                                                    one_hot(label, self.C),
                                                    norm_center_x.unsqueeze(1),
                                                    norm_center_y.unsqueeze(1),
                                                    norm_bndbox_w.unsqueeze(1),
                                                    norm_bndbox_h.unsqueeze(1)],
                                                   dim=1)

        img_tensor = fT.to_tensor(img)
        if self.normalize:
            mask_x, mask_y = mask
            fT.normalize(img_tensor[:, mask_y[0]:mask_y[1], mask_x[0]:mask_x[1]],
                         mean=self.normalize[0],
                         std=self.normalize[1],
                         inplace=True)

        return img_tensor, target


class ImgToTensor:
    """
    可调用图像张量转换类，将PIL图像转为张量，可选归一化。
    """

    def __init__(self, normalize: Optional[List] = None) -> None:
        """
        初始化归一化参数。

        :param normalize: 各通道的均值和标准差
        """
        self.normalize = normalize

    def __call__(self, sample: Tuple[Image.Image, List[Tuple[float, float]], th.Tensor]
                 ) -> Tuple[th.Tensor, th.Tensor]:
        """
        转换图像为张量并应用归一化。

        :param sample: 包含图像及其目标的元组
        :return: 图像张量及目标
        """
        img, mask, target = sample

        img_tensor = fT.to_tensor(img)
        if self.normalize:
            mask_x, mask_y = mask
            img_tensor[:, mask_y[0]:mask_y[1], mask_x[0]:mask_x[1]] = fT.normalize(img_tensor,
                                                                                   mean=self.normalize[0],
                                                                                   std=self.normalize[1])
        return img_tensor, target
        