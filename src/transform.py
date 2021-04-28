import random
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.ops.boxes import box_iou
from src.utils import Encoder


class SimpleTransformer(object):
    def __init__(self, dboxes, eval=False):
        self.eval = eval
        self.size = (300, 300)  # only support 300x300 ssd
        self.dboxes = dboxes
        self.encoder = Encoder(self.dboxes)
        self.crop = SSDCropping()
        self.hflip = RandomHorizontalFlip()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.img_trans = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
            transforms.ToTensor(),
            self.normalize
        ])
        self.trans_eval = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            self.normalize
        ])

    def __call__(self, img, bbox=None, label=None, max_num=200):
        if self.eval:
            bbox_out = torch.zeros(max_num, 4)
            label_out = torch.zeros(max_num, dtype=torch.long)
            bbox_out[:bbox.size(0), :] = bbox
            label_out[:label.size(0)] = label
            return self.trans_eval(img), bbox_out, label_out

        # img, img_size, bbox, label = self.crop(img, bbox, label)
        img, bbox = self.hflip(img, bbox)

        img = self.img_trans(img).contiguous()
        bbox, label = self.encoder.encode(bbox, label)

        return img, bbox, label


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, bboxes):
        if random.random() < self.prob:
            bboxes[:, 0], bboxes[:, 2] = 1.0 - bboxes[:, 2], 1.0 - bboxes[:, 0]
            return img.transpose(Image.FLIP_LEFT_RIGHT), bboxes
        return img, bboxes


class SSDCropping(object):
    def __init__(self):
        self.sample_options = (
            # Do nothing
            None,
            # min IoU, max IoU
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # no IoU requirements
            (None, None),
        )

    def __call__(self, img, bboxes, labels, img_size=(300, 300)):

        # Ensure always return cropped image
        while True:
            mode = random.choice(self.sample_options)

            if mode is None:
                return img, img_size, bboxes, labels

            htot, wtot = img_size

            min_iou, max_iou = mode
            min_iou = float("-inf") if min_iou is None else min_iou
            max_iou = float("+inf") if max_iou is None else max_iou

            for _ in range(1):
                w = random.uniform(0.3, 1.0)
                h = random.uniform(0.3, 1.0)

                if w / h < 0.5 or w / h > 2:
                    continue

                left = random.uniform(0, 1.0 - w)
                top = random.uniform(0, 1.0 - h)

                right = left + w
                bottom = top + h

                ious = box_iou(bboxes, torch.tensor([[left, top, right, bottom]]))

                if not ((ious > min_iou) & (ious < max_iou)).all():
                    continue

                # discard any bboxes whose center not in the cropped image
                xc = 0.5 * (bboxes[:, 0] + bboxes[:, 2])
                yc = 0.5 * (bboxes[:, 1] + bboxes[:, 3])

                masks = (xc > left) & (xc < right) & (yc > top) & (yc < bottom)

                # if no such boxes, continue searching again
                if not masks.any():
                    continue

                bboxes[bboxes[:, 0] < left, 0] = left
                bboxes[bboxes[:, 1] < top, 1] = top
                bboxes[bboxes[:, 2] > right, 2] = right
                bboxes[bboxes[:, 3] > bottom, 3] = bottom

                bboxes = bboxes[masks, :]
                labels = labels[masks]

                left_idx = int(left * wtot)
                top_idx = int(top * htot)
                right_idx = int(right * wtot)
                bottom_idx = int(bottom * htot)
                img = img.crop((left_idx, top_idx, right_idx, bottom_idx))

                bboxes[:, 0] = (bboxes[:, 0] - left) / w
                bboxes[:, 1] = (bboxes[:, 1] - top) / h
                bboxes[:, 2] = (bboxes[:, 2] - left) / w
                bboxes[:, 3] = (bboxes[:, 3] - top) / h

                htot = bottom_idx - top_idx
                wtot = right_idx - left_idx
                return img, (htot, wtot), bboxes, labels
