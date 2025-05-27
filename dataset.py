import os
import cv2
from torch.utils.data import Dataset
from albumentations import ElasticTransform
from albumentations.core.composition import Compose
import numpy as np

class Dermatology_data(Dataset):
    def __init__(self, Folder_img, Folder_mask, img_size = 320, mode = "train", p=0.5):
        self.Folder_img = Folder_img
        self.Folder_mask = Folder_mask
        self.mode = mode
        self.img_size = (img_size, img_size)
        self.data_paths = self.get_data_paths()
        if self.mode == "train":
            self.data_paths = self.data_paths[:int(len(self.data_paths)*0.8)]
        elif self.mode == "val":
            self.data_paths = self.data_paths[int(len(self.data_paths)*0.8):]
        else:
            self.data_paths = self.data_paths
        self.img_size = (img_size, img_size)
        self.images_masks = []
        self.p = p
        self.get_data()
    def __getitem__(self, index):
        if self.mode == "train" or self.mode == "val":
            img, mask = self.images_masks[index]
            # h, w, c to c, h, w
            img = np.transpose(img, (2, 0, 1))
            mask = np.transpose(mask, (2, 0, 1))
            return img, mask
        img = self.images_masks[index]
        img = np.transpose(img, (2, 0, 1))
        return img
    def __len__(self):
        return len(self.images_masks)
    def origin_len(self):
        return len(self.data_paths)
    def get_data_paths(self):
        data_paths = []
        if self.mode == "train" or self.mode == "val":
            for file_name in os.listdir(self.Folder_img):
                file_name = file_name.split(".")[0]
                data_paths.append([self.Folder_img + "/" + file_name + ".jpg",
                                   self.Folder_mask + "/" + file_name + ".png"])
        else:
            for file_name in os.listdir(self.Folder_img):
                file_name = file_name.split(".")[0]
                data_paths.append(self.Folder_img + "/" + file_name + ".jpg")
        return data_paths
    def get_data(self):
        for data_path in self.data_paths:
            if self.mode == "train" or self.mode == "val":
                # image = io.imread(data_path[0])
                # mask = io.imread(data_path[1])
                image = cv2.imread(data_path[0])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask  = cv2.imread(data_path[1])
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask[mask > 0] = 1
                # normalize mask to 0-1
                # mask = mask / 255.0
                # mask is a numpyndarray with shape (height, width), need to convert to (height, width, 1)
                if(image.shape[1] > self.img_size[0]):
                    image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_AREA) # tránh răng cưa
                    mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_AREA)
                    # image = resize(image, self.img_size, anti_aliasing=True) # tránh răng cưa
                    # mask = resize(mask, self.img_size, anti_aliasing=True)
                    #
                    if np.random.uniform(0,1) >= self.p:
                        transform = Compose([
                            ElasticTransform(alpha=120, sigma=12, p=1.0)
                        ])
                        augmented = transform(image=image, mask=mask)
                        image_aug = augmented["image"]
                        mask_aug = augmented["mask"]
                        # add last dim to mask
                        mask_aug = np.expand_dims(mask_aug, axis=-1)
                        self.images_masks.append([image_aug, mask_aug])
                    # add last dim to mask
                    mask = np.expand_dims(mask, axis=-1)
                    self.images_masks.append([image, mask])
                else:
                    image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_CUBIC) # sử dụng bicubic nội suy tính giá trị pixel mới
                    mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_CUBIC)
                    # image = resize(image, self.img_size, order=3) # sử dụng bicubic nội suy tính giá trị pixel mới
                    # mask = resize(mask, self.img_size, order=3)
                    if np.random.uniform(0,1) >= self.p:
                        transform = Compose([
                            ElasticTransform(alpha=120, sigma=12, p=1.0)
                        ])
                        augmented = transform(image=image, mask=mask)
                        image_aug = augmented["image"]
                        mask_aug = augmented["mask"]
                        # add last dim to mask
                        mask_aug = np.expand_dims(mask_aug, axis=-1)
                        self.images_masks.append([image_aug, mask_aug])
                    # add last dim to mask
                    mask = np.expand_dims(mask, axis=-1)
                    self.images_masks.append([image, mask])
            else:
                # image = io.imread(data_path[0])
                image = cv2.imread(data_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if(image.shape[1] > self.img_size[0]):
                    image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_AREA) # tránh răng cưa
                else:
                    image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_CUBIC) # sử dụng bicubic nội suy tính giá trị pixel mới
                self.images_masks.append(image)
