import os
import pickle
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import cv2
from skimage.feature import canny
import skimage.draw
from skimage.color import rgb2gray


def to_int(x):
    return tuple(map(int, x))


class SketchDataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, augment=True, aug_for_alias=True, training=True):
        super(SketchDataset, self).__init__()
        self.augment = augment
        self.training = training

        f = open(flist, 'r')
        self.data = [d.strip() for d in f.readlines()]
        f.close()
        self.wireframe_path = config.wireframe_path
        self.wireframe_th = config.wireframe_th

        self.input_size = config.input_size
        self.output_size = config.output_size
        self.aug_for_alias = aug_for_alias

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_edge(self, img, sigma):
        return canny(img, sigma=sigma, mask=None).astype(np.float)

    def load_wireframe(self, idx, size):
        selected_img_name = self.data[idx]
        line_name = selected_img_name.split("/")[-1]
        line_name = self.wireframe_path + line_name.replace('.png', '.pkl').replace('.jpg', '.pkl')
        wf = pickle.load(open(line_name, 'rb'))
        lmap = np.zeros((size, size))
        for i in range(len(wf['scores'])):
            if wf['scores'][i] > self.wireframe_th:
                line = wf['lines'][i].copy()
                line[0] = line[0] * size
                line[1] = line[1] * size
                line[2] = line[2] * size
                line[3] = line[3] * size
                rr, cc, value = skimage.draw.line_aa(*to_int(line[0:2]), *to_int(line[2:4]))
                lmap[rr, cc] = np.maximum(lmap[rr, cc], value)

        return lmap

    def load_item(self, index):
        # load image
        img = Image.open(self.data[index]).convert("RGB")
        img = np.array(img)
        # resize/crop if needed
        img_large = self.resize(img, self.output_size, self.output_size)
        # augment data
        if self.augment and random.random() > 0.5 and self.training:
            img_large = img_large[:, ::-1, ...]
        if self.augment and random.random() > 0.5 and self.training:
            img_large = img_large[::-1, :, ...]

        gray_large = rgb2gray(img_large)
        edge_large = self.load_edge(gray_large, sigma=2.5)
        line_large = self.load_wireframe(index, self.output_size)
        # line_large[line_large >= 0.5] = 1
        # line_large[line_large < 0.5] = 0

        img_small = self.resize(img, self.input_size, self.input_size)
        gray_small = rgb2gray(img_small)
        edge_small = self.load_edge(gray_small, sigma=2.0)
        line_small = self.load_wireframe(index, self.input_size)
        line_small_gt = line_small.copy()
        if random.random() < 0.5 and self.aug_for_alias:
            line_small[line_small >= 0.5] = 1
            line_small[line_small < 0.5] = 0

        batch = {'name': self.load_name(index),
                 'edge_small': self.to_tensor(edge_small),
                 'line_small': self.to_tensor(line_small),
                 'line_small_gt': self.to_tensor(line_small_gt),
                 'edge_large': self.to_tensor(edge_large),
                 'line_large': self.to_tensor(line_large)}

        return batch

    def to_tensor(self, img):
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, center_crop=False):
        imgh, imgw = img.shape[0:2]

        if center_crop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        if imgh > height and imgw > width:
            inter = cv2.INTER_AREA
        else:
            inter = cv2.INTER_LINEAR
        img = cv2.resize(img, (height, width), interpolation=inter)

        return img

    def load_flist(self, flist):

        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = self.getfilelist(flist)
                flist.sort(key=lambda x: x.split('/')[-1])
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

    def getfilelist(self, path):
        all_file = []
        for dir, folder, file in os.walk(path):
            for i in file:
                t = "%s/%s" % (dir, i)
                if t.endswith('.png') or t.endswith('.jpg') or t.endswith('.JPG') \
                        or t.endswith('.PNG') or t.endswith('.JPEG'):
                    all_file.append(t)
        return all_file
