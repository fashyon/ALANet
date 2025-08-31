import os.path
from os.path import join
from data.transforms import to_tensor, ReflectionSythesis_1
from PIL import Image
import random
import torch
import math
import torchvision.transforms.functional as F
from util.util import  pre_caption
import data.torchdata as torchdata
import numpy as np
import json
from torch.utils.data import Dataset,DataLoader

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)

def __scale_height(img, target_height):
    ow, oh = img.size
    if (oh == target_height):
        return img
    h = target_height
    w = int(target_height * ow / oh)
    w = math.ceil(w / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)


def paired_data_transforms(img_1, img_2, unaligned_transforms=False):
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw  # Return the starting position of the crop (i, j) and the crop height and width (th, tw)

    target_size = int(random.randint(224, 448) / 2.) * 2
    ow, oh = img_1.size
    if ow >= oh: # Scale based on the shorter side, making the shorter side equal to the target size. target_size is a randomly generated value between 224 and 448.
        img_1 = __scale_height(img_1, target_size)
        img_2 = __scale_height(img_2, target_size)
    else:
        img_1 = __scale_width(img_1, target_size)
        img_2 = __scale_width(img_2, target_size)

    if random.random() < 0.5:
        img_1 = F.hflip(img_1)
        img_2 = F.hflip(img_2)

    i, j, h, w = get_params(img_1, (224,224)) # Crop to (224, 224) size
    img_1 = F.crop(img_1, i, j, h, w)
    
    if unaligned_transforms:
        i_shift = random.randint(-10, 10)
        j_shift = random.randint(-10, 10)
        i += i_shift
        j += j_shift

    img_2 = F.crop(img_2, i, j, h, w)
    
    return img_1,img_2



BaseDataset = torchdata.Dataset


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, *args, **kwargs)
        self.shuffle = shuffle

    def reset(self):
        if self.shuffle:
            print('Reset Dataset...')
            self.dataset.reset()

class CEILDataset_Clip(BaseDataset):
    def __init__(self, datadir,lines=None, size=None, enable_transforms=True, low_sigma=2, high_sigma=5, low_gamma=1.3,
                 high_gamma=1.3,target_r_prob=1,caption_t_prob=1,caption_r_prob=1):
        super(CEILDataset_Clip, self).__init__()
        self.size = size
        self.datadir = datadir
        self.enable_transforms = enable_transforms
        self.syn_model = ReflectionSythesis_1(kernel_sizes=[11], low_sigma=low_sigma, high_sigma=high_sigma,
                                              low_gamma=low_gamma, high_gamma=high_gamma)
        self.lines = lines
        self.reset(shuffle=False)
        self.target_r_prob = target_r_prob
        self.caption_t_prob = caption_t_prob
        self.caption_r_prob = caption_r_prob

    def reset(self, shuffle=True):
        if shuffle:
            random.shuffle(self.lines)

        num_paths = len(self.lines) // 2
        # Transmission images
        self.T_paths = self.lines[0:num_paths]
        self.text_T = []
        self.image_T = []
        self.txt2img_T = {}  # txt2img for transmission images
        self.img2txt_T = {}  # img2txt for transmission images
        txt_id_T = 0  # txt_id for transmission images

        # Reflection images
        self.R_paths = self.lines[num_paths:2 * num_paths]
        self.text_R = []
        self.image_R = []
        self.txt2img_R = {}  # txt2img for reflection images
        self.img2txt_R = {}  # img2txt for reflection images
        txt_id_R = 0  # txt_id for reflection images

        # Process transmission images
        for img_id, ann in enumerate(self.T_paths):
            self.image_T.append(join(self.datadir,ann['image']))
            self.img2txt_T[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text_T.append(pre_caption(caption, 77))
                self.img2txt_T[img_id].append(txt_id_T)
                self.txt2img_T[txt_id_T] = img_id
                txt_id_T += 1

        # Process reflection images
        for img_id, ann in enumerate(self.R_paths):
            self.image_R.append(join(self.datadir,ann['image']))
            self.img2txt_R[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text_R.append(pre_caption(caption, 77))
                self.img2txt_R[img_id].append(txt_id_R)
                self.txt2img_R[txt_id_R] = img_id
                txt_id_R += 1

    def data_synthesis(self, t_img, r_img):
        if self.enable_transforms:
            t_img, r_img = paired_data_transforms(t_img, r_img)
        syn_model = self.syn_model
        t_img, r_img, m_img = syn_model(t_img, r_img)

        T = to_tensor(t_img)
        R = to_tensor(r_img)
        M = to_tensor(m_img)

        return T, R, M

    def __getitem__(self, index):
        index_T = index % len(self.image_T)
        index_R = index % len(self.image_R)

        T_path = self.image_T[index_T]
        R_path = self.image_R[index_R]
        t_img = Image.open(T_path).convert('RGB')
        r_img = Image.open(R_path).convert('RGB')

        T, R, M = self.data_synthesis(t_img, r_img)

        fn = os.path.basename(T_path)

        # Randomly decide whether to keep target_r, caption_t, and caption_r
        if random.random() < self.target_r_prob:
            target_r = R
        else:
            target_r = None

        if random.random() < self.caption_t_prob:
            if self.img2txt_T[index_T]:
                caption_t = self.text_T[np.random.choice(self.img2txt_T[index_T])]
            else:
                caption_t = None
        else:
            caption_t = None

        if random.random() < self.caption_r_prob:
            if self.img2txt_R[index_R]:
                caption_r = self.text_R[np.random.choice(self.img2txt_R[index_R])]
            else:
                caption_r = None
        else:
            caption_r = None

       # Create a dictionary with non-None values only
        return {'input': M, 'target_t': T, 'fn': fn, 'real': False, **{k: v for k, v in {'target_r': target_r, 'caption_t': caption_t, 'caption_r': caption_r}.items() if v is not None}}

    def __len__(self):
        if self.size is not None:
            return min(max(len(self.T_paths), len(self.R_paths)), self.size)
        else:
            return max(len(self.T_paths), len(self.R_paths))

class CEILDataset_Clip_Real(BaseDataset):
    def __init__(self, datadir, json_file=None,exist_R_caption= False, enable_transforms=False, unaligned_transforms=False, target_r_prob=1,
                 caption_t_prob=1, caption_r_prob=1):
        super(CEILDataset_Clip_Real, self).__init__()
        self.datadir = datadir
        self.paths = json.load(open(join(self.datadir, json_file), mode='r', encoding='utf-8'))
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.datadir_M = join(self.datadir, 'blended')
        self.datadir_T = join(self.datadir, 'transmission_layer')
        self.datadir_R = join(self.datadir, 'reflection_layer')
        # Check if the path contains files
        self. exist_R_caption =exist_R_caption
        if self. exist_R_caption ==True:
            print(str(self.datadir) + ' contains Reflection captions!')
        else:
            print(str(self.datadir) + ' does not contain Reflection captions!')

        # Define probabilities for target_r, caption_t, and caption_r
        self.target_r_prob = target_r_prob
        self.caption_t_prob = caption_t_prob
        self.caption_r_prob = caption_r_prob

        # Transmission images
        self.image = []
        self.text_T = []
        self.txt2img_T = {}  # txt2img for transmission images
        self.img2txt_T = {}  # img2txt for transmission images
        txt_id_T = 0  # txt_id for transmission images

        # Reflection images
        self.text_R = []
        self.txt2img_R = {}  # txt2img for reflection images
        self.img2txt_R = {}  # img2txt for reflection images
        txt_id_R = 0  # txt_id for reflection images

        for img_id, ann in enumerate(self.paths):
            self.image.append(ann['image'])
            self.img2txt_T[img_id] = []
            for i, caption in enumerate(ann['caption_T']):
                self.text_T.append(pre_caption(caption, 77))
                self.img2txt_T[img_id].append(txt_id_T)
                self.txt2img_T[txt_id_T] = img_id
                txt_id_T += 1
            self.img2txt_R[img_id] = []
            if self.exist_R_caption:
                for i, caption in enumerate(ann['caption_R']):
                    self.text_R.append(pre_caption(caption, 77))
                    self.img2txt_R[img_id].append(txt_id_R)
                    self.txt2img_R[txt_id_T] = img_id
                    txt_id_R += 1

        # Check if the number of files matches the number of images
        self.num_images = len(self.image)
        num_files = len(os.listdir(self.datadir_M))
        if self.num_images != num_files:
            raise ValueError(
                f"Number of files in {self.datadir_M} does not match the number of images: {num_files} vs {self.num_images}")

    def __getitem__(self, index):
        fn = self.image[index]
        m_img = Image.open(join(self.datadir_M, fn)).convert('RGB')
        t_img = Image.open(join(self.datadir_T, fn)).convert('RGB')

        if self.enable_transforms:
            t_img, m_img = paired_data_transforms(t_img, m_img, self.unaligned_transforms)

        if random.random() < self.caption_t_prob:
            caption_t = self.text_T[np.random.choice(self.img2txt_T[index])]
        else:
            caption_t = None

        if random.random() < self.caption_r_prob:
            caption_r = self.text_R[np.random.choice(self.img2txt_R[index])]
        else:
            caption_r = None

        input = to_tensor(m_img)
        target_t = to_tensor(t_img)

        # Create a dictionary with non-None values only
        return {'input': input, 'target_t': target_t, 'fn': fn,
                **{k: v for k, v in {'caption_t': caption_t, 'caption_r': caption_r}.items() if
                   v is not None}}

    def __len__(self):
        return self.num_images

class CEILTestDataset_NoLanguage(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=False, unaligned_transforms=False):
        super(CEILTestDataset_NoLanguage, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir, 'blended'))
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms

        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]

        t_img = Image.open(join(self.datadir, 'transmission_layer', fn)).convert('RGB')
        m_img = Image.open(join(self.datadir, 'blended', fn)).convert('RGB')

        if self.enable_transforms:
            t_img, m_img = paired_data_transforms(t_img, m_img, self.unaligned_transforms)

        T = to_tensor(t_img)
        M = to_tensor(m_img)

        dic = {'input': M, 'target_t': T, 'fn': fn}
        return dic

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)

class CEILTestDataset_Clip(BaseDataset):
    def __init__(self, datadir, json_file=None,exist_R_caption= False, target_r_prob=1,
                 caption_t_prob=1, caption_r_prob=1):
        super(CEILTestDataset_Clip, self).__init__()
        self.datadir = datadir
        self.paths = json.load(open(join(self.datadir, json_file), mode='r', encoding='utf-8'))
        self.datadir_M = join(self.datadir, 'blended')
        self.datadir_T = join(self.datadir, 'transmission_layer')
        self.datadir_R = join(self.datadir, 'reflection_layer')

        self.exist_R_caption = exist_R_caption
        if self.exist_R_caption == True:
            print(str(self.datadir) + ' contains Reflection captions!')
        else:
            print(str(self.datadir) + ' does not contain Reflection captions!')

        # Check if the path contains files
        if os.path.exists(self.datadir_R) and os.listdir(self.datadir_R):
            self.exist_R_img = True
            print(str(self.datadir) + ' contains Reflection images!')
        else:
            self.exist_R_img = False
            print(str(self.datadir) + ' does not contain Reflection images!')

        # Define probabilities for target_r, caption_t, and caption_r
        self.target_r_prob = target_r_prob
        self.caption_t_prob = caption_t_prob
        self.caption_r_prob = caption_r_prob

        # Transmission images
        self.image = []
        self.text_T = []
        self.txt2img_T = {}  # txt2img for transmission images
        self.img2txt_T = {}  # img2txt for transmission images
        txt_id_T = 0  # txt_id for transmission images

        # Reflection images
        self.text_R = []
        self.txt2img_R = {}  # txt2img for reflection images
        self.img2txt_R = {}  # img2txt for reflection images
        txt_id_R = 0  # txt_id for reflection images


        for img_id, ann in enumerate(self.paths):
            self.image.append(ann['image'])
            self.img2txt_T[img_id] = []
            for i, caption in enumerate(ann['caption_T']):
                self.text_T.append(pre_caption(caption, 77))
                self.img2txt_T[img_id].append(txt_id_T)
                self.txt2img_T[txt_id_T] = img_id
                txt_id_T += 1
            self.img2txt_R[img_id] = []
            if self.exist_R_caption:
                for i, caption in enumerate(ann['caption_R']):
                    self.text_R.append(pre_caption(caption, 77))
                    self.img2txt_R[img_id].append(txt_id_R)
                    self.txt2img_R[txt_id_T] = img_id
                    txt_id_R += 1

        # Check if the number of files matches the number of images
        self.num_images = len(self.image)
        num_files = len(os.listdir(self.datadir_M))
        if self.num_images != num_files:
            raise ValueError(
                f"Number of files in {self.datadir_M} does not match the number of images: {num_files} vs {self.num_images}")

    def __getitem__(self, index):
        fn = self.image[index]
        m_img = Image.open(join(self.datadir_M, fn)).convert('RGB')
        t_img = Image.open(join(self.datadir_T, fn)).convert('RGB')

        caption_t = self.text_T[np.random.choice(self.img2txt_T[index])]
        caption_r = None
        target_r = None
        if self.exist_R_caption == True:
            caption_r = self.text_R[np.random.choice(self.img2txt_R[index])]
            caption_r = caption_r if random.random() < self.caption_r_prob else None
            if self.exist_R_img == True:
                r_img = Image.open(join(self.datadir_R, fn)).convert('RGB')
                target_r = to_tensor(r_img) if random.random() < self.target_r_prob else None

        input = to_tensor(m_img)
        target_t = to_tensor(t_img)

        # Randomly decide whether to keep caption_t and caption_r
        caption_t = caption_t if random.random() < self.caption_t_prob else None

        # Create a dictionary with non-None values only
        return {'input': input, 'target_t': target_t, 'fn': fn,
                **{k: v for k, v in {'target_r': target_r, 'caption_t': caption_t, 'caption_r': caption_r}.items() if
                   v is not None}}

    def __len__(self):
        return self.num_images


class FusionDataset(BaseDataset):
    def __init__(self, datasets, fusion_ratios=None):
        self.datasets = datasets
        self.size = sum([len(dataset) for dataset in datasets])
        self.fusion_ratios = fusion_ratios or [1./len(datasets)] * len(datasets)
        print('[i] using a fusion dataset: %d %s imgs fused with ratio %s'%(self.size, [len(dataset) for dataset in datasets], self.fusion_ratios))

    def reset(self):
        for dataset in self.datasets:
            dataset.reset()

    def __getitem__(self, index):
        residual = 1
        for i, ratio in enumerate(self.fusion_ratios):
            if random.random() < ratio/residual or i == len(self.fusion_ratios) - 1:
                dataset = self.datasets[i]
                return dataset[index%len(dataset)]
            residual -= ratio
    
    def __len__(self):
        return self.size

