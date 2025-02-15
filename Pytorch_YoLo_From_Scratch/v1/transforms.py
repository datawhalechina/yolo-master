import torch as th
from torch.nn.functional import one_hot
import torchvision.transforms.functional as fT
from PIL.Image import Image
from typing import Tuple, Optional, List


class Resize:
    """
    A callable Resize class, which upon its call resizes the image and scales the bounding box coordinates
    appropriately.
    """

    def __init__(self, output_size: int) -> None:
        """
        Initialize the dimension d of the image after the transformation. After the image is resized, it will have a
        (d x d) shape.

        :param output_size: The dimension of the image after the transformation.
        """
        self.d = output_size

    def __call__(self, sample: Tuple[Image, th.Tensor]
                 ) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        """
        Resize the image to a (d x d) shape and transform the bounding box coordinates.
        In an image with N objects, the target tensor has a (N x 5)-shape and for each object the target is formatted
        as  <classification_id>, <x_min>, <y_min>, <x_max>, <y_max>. Given an (h x w)-image, the x and y coordinates
        are updated to x' and y' in the following way:

        | x' = x * d / w
        | y' = y * d / h

        :param sample: A tuple containing the image and its corresponding target
        :return: The resized (d x d) image, a mask that contains all the image pixels ([0,d] both in the x- and y-axis)
                 and the appropriately scaled coordinates
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
    A callable RandomScaleTranslate class, which resizes the image and scales the bounding box coordinates. In order to
    augment the dataset, for each image we randomly choose between the following operations:

    - resize
    - zoom out & resize
    - zoom in & resize

    When we zoom out, the image will be padded with zeros. To avoid distorting these zero values
    (e.g. RandomColorJitter, normalization), a mask is returned to specify which values were padded.
    """
    def __init__(self,
                 output_size: int,
                 jitter: float,
                 resize_p: float,
                 zoom_out_p: float,
                 zoom_in_p: float) -> None:
        """
        Initialize the dimension d of the image after the transformation. After the image is resized, it will have a
        (d x d) shape. The given jitter factor is also stored to randomly scale and translate the image.
        The probabilities are used to select randomly one of the operations.

        :param output_size: The dimension of the image after the transformation.
        :param jitter: A factor to sample the random scale and translation for the zoom operations
        :param resize_p: The probability that the 'resize' operation is applied
        :param zoom_out_p: The probability that the 'zoom out & resize' operation is applied
        :param zoom_in_p: The probability that the 'zoom in & resize' operation is applied
        """
        self.d = output_size
        self.jitter = jitter
        self.t_probs = th.cumsum(th.Tensor([resize_p, zoom_out_p, zoom_in_p]), dim=0)

    def __call__(self, sample: Tuple[Image, th.Tensor]
                 ) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        """
        Sample from a uniform random distribution whether to apply the 'resize', 'zoom out & resize' or 'zoom in &
        resize' operation. The probability of each operation is equal to the given corresponding value.

        In each case, the image is resized to a (d x d) shape and the bounding box coordinates are transformed
        appropriately. A mask that specifies the bounds of the non-padded values of the image is also returned.
        For the 'resize' and the 'zoom in & resize' operations, this mask contains all the pixel values of the image.

        If a bounding box is very small after the transformation, it is removed from the targets.

        :param sample: A tuple containing the image and its corresponding target
        :return: A tuple containing the transformed image, its mask and the updated corresponding target
        """

        transform_prob = th.rand(1)
        if transform_prob < self.t_probs[0]:                    # resize
            img, mask, target = self._resize(sample)
        elif transform_prob < self.t_probs[1]:                  # zoom out & resize
            img, mask, target = self._zoom_out(sample)
        else:                                                   # zoom in & resize
            img, mask, target = self._zoom_in(sample)

        # Remove very small bounding boxes
        bboxes_w = target[:, 3] - target[:, 1]
        bboxes_h = target[:, 4] - target[:, 2]
        threshold = 0.001 * self.d
        valid_bboxes = th.logical_not(th.logical_or(bboxes_w < threshold, bboxes_h < threshold))
        target = target[valid_bboxes]
        return img, mask, target

    def _resize(self, sample: Tuple[Image, th.Tensor]
                ) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        """
        This function follows the same logic with the __call__ function of the Resize class.

        Resize the image to a (d x d) shape and transform the bounding box coordinates.
        In an image with N objects, the target tensor has a (N x 5)-shape and for each object the target is formatted
        as  <classification_id>, <x_min>, <y_min>, <x_max>, <y_max>. Given an (h x w)-image, the x and y coordinates
        are updated to x' and y' in the following way:

        | x' = x * d / w
        | y' = y * d / h

        :param sample: A tuple containing the image and its corresponding target
        :return: A tuple containing the resized image, its mask and the updated corresponding target
        """
        img, target = sample
        w, h = img.size

        img = fT.resize(img, (self.d, self.d))
        target[:, [1, 3]] *= self.d / w
        target[:, [2, 4]] *= self.d / h

        mask = [(0, self.d), (0, self.d)]
        return img, mask, target

    def _zoom_out(self, sample: Tuple[Image, th.Tensor]
                  ) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        """
        First a new aspect ratio is set by sampling randomly from a uniform distribution rand_w and rand_h:

        - rand_w ~ U((1-jitter)w, (1+jitter)w)
        - rand_h ~ U((1-jitter)h, (1+jitter)h)

        and setting:
         new_ar = rand_w / rand_h

        We compare rand_w with rand_h and set the large dimension's size equal with d. The size of the other dimension
        is calculated based on the aspect ratio. Therefore, the selected image patch has a size of (d, k) or (k,d) with
        k <= 1.

        Following this resize transformation, the image patch is randomly translated. To translate the image patch, we
        pad the image with zeros:
        - left and right, if the image patch has a width of k
        - top and bottom, if the image patch has a height of k.

        We randomly sample how many pixels are padded on the left or the top of the image from U(0, d-k). We also pad
        the image with zeros on the right or the bottom to have a (d x d) shape.

        The transformations that are applied to the coordinates of the image are:
        1) resize from (w,h) to (d,k) or (k,d)
        2) translate the image by the number of padded values on the left or the top of the image
        Therefore, these transformations will be applied to the bounding box coordinates.

        The mask of the transformed image will contain the bounds of the non-padded values with mask = [mask_x, mask_y]

        :param sample: A tuple containing the image and its corresponding target
        :return: A tuple containing the transformed image after the 'zoom out & resize' operation, its mask and the
                 updated corresponding target
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

    def _zoom_in(self, sample: Tuple[Image, th.Tensor]
                 ) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        """
        First we sample the width and height of an image patch, nw and nh respectively, from a uniform random
        distribution:

        - nw ~ U((1-jitter)w, w)
        - nh ~ U((1-jitter)h, h)

        Similarly we sample dx and dy to crop an image patch from the original image.

        - dx ~ U(0, w-nw)
        - dy ~ U(0, h-nh)

        Following that, the selected image patch is resized to a (d x d) shape.

        The bounding box coordinates are transformed in the following way:
        1) the top, left coordinate of the image patch (dx, dy) must be translated to (0,0)
        2) the image patch is resized from a size of (nw, nh) to (d, d)
        The bounding boxes that are not visible after the transformation are completely removed from the targets, while
        the bounding boxes that are only partially visible have their coordinates clamped to be within the image.

        The mask of the transformed image will contain all the pixel values of the image in both the x- and y-axis.

        :param sample: A tuple containing the image and its corresponding target
        :return: A tuple containing the transformed image after the 'zoom in & resize' operation, its mask and the
                 updated corresponding target
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

        # Remove bounding boxes that are not visible any more
        target = target[th.logical_not(th.logical_or(th.logical_or(target[:, 3] < 0, target[:, 1] > self.d),
                                                     th.logical_or(target[:, 4] < 0, target[:, 2] > self.d)))]

        # Update the bounds of the bounding boxes that are only partially visible
        target[:, [1, 2]] = target[:, [1, 2]].clamp(min=0)
        target[:, [3, 4]] = target[:, [3, 4]].clamp(max=self.d)

        mask = [(0, self.d), (0, self.d)]
        return img, mask, target


class RandomColorJitter:
    """
    A callable RandomColorJitter class, which when called distorts the colors of the input image. The target values
    remain unchanged.
    """

    def __init__(self, hue: float, sat: float, exp: float):
        """
        Initialize the hue, saturation and exposure parameters.

        :param hue: The hue parameter. The hue value will be sampled uniformly at random from [-hue, hue].
        :param sat: The saturation parameter. The saturation value will be sampled uniformly at random from [1/sat, sat]
        :param exp: The exposure parameter. The exposure parameter will be sampled uniformly at random from [1/exp, exp]
        """
        self.hue = hue
        self.sat = sat
        self.exp = exp

    def __call__(self, sample: Tuple[Image, List[Tuple[float, float]], th.Tensor]
                 ) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        """
        Sample uniformly at random the hue, saturation and exposure values and distort the colors of the input image.
        The hue, saturation and exposure of the image are adjusted in the HSV color space. Specifically:

        HUE
            pixel_H = pixel_H + rand_hue

            if pixel_H > 1, then
                pixel_H = pixel_H - 1
            else if pixel_H < 0, then
                pixel_H = pixel_H + 1

        SATURATION
            pixel_S = min(pixel_S * rand_sat, 1.0)

        EXPOSURE
            pixel_V = min(pixel_V * rand_exp, 1.0)

        :param sample: A tuple containing the image, its mask and the corresponding target
        :return: The distorted image and its (unchanged) target
        """
        # Sample uniformly at random the hue, saturation and exposure values for this image.
        rand_hue = th.Tensor(1).uniform_(-self.hue, self.hue)
        rand_sat = th.Tensor(1).uniform_(1 / self.sat, self.sat)
        rand_exp = th.Tensor(1).uniform_(1 / self.exp, self.exp)

        # Convert the RGB PIL image to an HSV tensor.
        rgb_img, mask, target = sample
        hsv_img = rgb_img.convert('HSV')
        hsv_tensor = fT.to_tensor(hsv_img)

        mask_x, mask_y = mask
        masked_hsv_tensor = hsv_tensor[:, mask_y[0]:mask_y[1], mask_x[0]:mask_x[1]]

        # Adjust hue
        masked_hsv_tensor[0, :, :] += rand_hue
        masked_hsv_tensor[0, :, :] += (1. * (masked_hsv_tensor[0, :, :] < 0) - 1. * \
                                       (masked_hsv_tensor[0, :, :] > 1)) * th.ones_like(masked_hsv_tensor[0, :, :])
        # Adjust saturation
        masked_hsv_tensor[1, :, :] *= rand_sat
        masked_hsv_tensor[1, :, :] = masked_hsv_tensor[1, :, :].clamp(max=1.0)

        # Adjust exposure
        masked_hsv_tensor[2, :, :] *= rand_exp
        masked_hsv_tensor[2, :, :] = masked_hsv_tensor[2, :, :].clamp(max=1.0)

        # Convert the HSV tensor to an RGB PIL image
        hsv_img = fT.to_pil_image(hsv_tensor, mode='HSV')
        rgb_img = hsv_img.convert('RGB')

        return rgb_img, mask, target


class RandomHorizontalFlip:
    """
    A callable RandomHorizontalFlip class. When called, it is randomly chosen whether the image is flipped horizontally.
    When the image is  flipped, the bounding box coordinates and the mask are also transformed appropriately.
    """
    def __init__(self, p: float) -> None:
        """
        Initialize a RandomHorizontalFlip object and set the probability that the image is flipped.

        :param p: The probability that the horizontal flip transformation is applied
        """
        self.p = p

    def __call__(self, sample: Tuple[Image, List[Tuple[float, float]], th.Tensor]
                 ) -> Tuple[Image, List[Tuple[float, float]], th.Tensor]:
        """
        A number in [0,1) is randomly sampled from the uniform distribution U(0,1) to determine if the horizontal flip
        transformation will be applied. The transformation is applied with probability p. If the image is flipped, the
        xmin and xmax coordinates of the bounding boxes are updated. Furthermore, the mask's component in the x-axis
        is also updated similarly.

        :param sample: A tuple containing the image, its mask and the corresponding target
        :return: If the transformation is applied, the horizontally flipped image, the transformed mask and the
                 transformed target is returned. Otherwise, the input sample is returned.
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
    A callable ToYOLOTensor class. When called the targets of the image will be transformed according to the YOLO
    format, while the PIL image will be converted a Tensor. If the mean and the standard deviation of the input image
    channels are provided, the Tensors are normalized.
    """

    def __init__(self, S: int, C: int, normalize: Optional[List] = None) -> None:
        """
        Initialize the number of grid cells per row/column and the number of classes of the dataset.

        :param S: The S parameter of the YOLO algorithm. Each image is split into an (S x S) grid.
        :param C: The number of classes of the dataset.
        :param normalize: A list that contains two lists, one with the 3 mean values of the pixels per channel and
                          another with the corresponding standard deviations per channel.
        """
        self.S = S
        self.C = C
        self.normalize = normalize

    def __call__(self, sample: Tuple[Image, List[Tuple[float, float]], th.Tensor]
                 ) -> Tuple[th.Tensor, th.Tensor]:
        """
        The PIL image input is converted to a Tensor and the tensor is (optionally normalized).
        In an image with N objects, the input target tensor has a (N x 5)-shape and for each object the target is
        formatted as <classification_id>, <x_min>, <y_min>, <x_max>, <y_max>. The output target tensor has shape
        (S x S x C+5). For each of the (S x S) cells of the grid:

        - index 0: 0 or 1 if an object exists in that cell
        - indices [1,C]: one hot representation of the object in the cell or 0s everywhere
        - index C+1: normalized center x-coordinate.
        - index C+2: normalized center y-coordinate.
        - index C+3: normalized width of the bounding box
        - index C+4: normalized height of the bounding box

        The center coordinates are normalized as offsets in the grid where the upper-left corner in the grid has
        coordinates (0,0) and the bottom-right corner in the grid has coordinates (1,1).

        The height and the width of the bounding boxes are normalized by the image height and width.

        :param sample: A tuple containing the image, its mask and the corresponding target
        :return: The given image and its target in a YOLO-grid format.
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
        A callable ImgToTensor class. When called the PIL image will be converted a Tensor. If the mean and the standard
        deviation of the input image channels are provided, the Tensors are normalized.
        """

    def __init__(self, normalize: Optional[List] = None) -> None:
        """
        Initialize the number of grid cells per row/column and the number of classes of the dataset.

        :param S: The S parameter of the YOLO algorithm. Each image is split into an (S x S) grid.
        :param C: The number of classes of the dataset.
        :param normalize: A list that contains two lists, one with the 3 mean values of the pixels per channel and
                          another with the corresponding standard deviations per channel.
        """
        self.normalize = normalize

    def __call__(self, sample: Tuple[Image, List[Tuple[float, float]], th.Tensor]
                 ) -> Tuple[th.Tensor, th.Tensor]:
        """
        The PIL image input is converted to a Tensor and the tensor is (optionally normalized). The targets are not
        modified.

        :param sample: A tuple containing the image and its corresponding target
        :return: An image tensor and the corresponding targets.
        """
        img, mask, target = sample

        img_tensor = fT.to_tensor(img)
        if self.normalize:
            mask_x, mask_y = mask
            img_tensor[:, mask_y[0]:mask_y[1], mask_x[0]:mask_x[1]] = fT.normalize(img_tensor,
                                                                                   mean=self.normalize[0],
                                                                                   std=self.normalize[1])
        return img_tensor, target
