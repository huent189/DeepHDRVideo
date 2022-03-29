"""
Dataloader for the captured real-world dataset
It supports static scene with GT, dynamic scene with GT, and dynamic without GT
"""
import os
import numpy as np
from imageio import imread
import torch, glob, random
import torch.utils.data as data

from datasets import hdr_transforms
from datasets.tog13_online_align_dataset import tog13_online_align_dataset
from utils import utils
import utils.image_utils as iutils
np.random.seed(0)

class train_real_dataset(tog13_online_align_dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, args, split='train'):
        self.root  = os.path.join(args.data_dir)
        self.split = split
        self.args  = args
        self.nframes = args.nframes
        self.expo_num = args.nexps if hasattr(args, 'nexps') else 2
        self._prepare_list()
        self.run_model = args.run_model if hasattr(args, 'run_model') else False
        #print(self.expos_list)
    
    def _prepare_list(self): 
        suffix = '' if self.args.test_scene is None else self.args.test_scene
        self.scene_list = glob.glob(os.path.join(self.root, '*/'))
        self.scene_list.sort()

        skip_headtail = False 
        if self.nframes == 3 or (self.nframes == 5 and self.expo_num == 3):
            skip_headtail = True

        self._collect_triple_list(self.scene_list)
        print('[%s] totaling  %d triples' % (self.__class__.__name__, len(self.img_list)))
    def _collect_triple_list(self, scene_list):
        self.expos_list = []
        self.img_list = []
        sample_len = self.nframes * 2 - 1
        self.total_img_num = 0
        for i in range(len(scene_list)):
            img_dir = os.path.join(self.root, scene_list[i])
            img_list = glob.glob(os.path.join(img_dir, '*.jpg'))
            img_list.sort()
            img_list =[os.path.join(img_dir, img_path) for img_path in img_list]
            e_list = self._load_exposure_list(os.path.join(img_dir, 'Exposures.txt'), img_num=len(img_list))
            self.expos_list += [e_list[i:i+sample_len] for i in range(len(e_list) - sample_len)]
            self.img_list += [img_list[i:i+sample_len] for i in range(len(img_list) - sample_len)]
            
    def _get_input_path(self, index):
        return self.img_list[index], self.expos_list[index]
    def read_im(self, p):
        if p[-4:] == '.tif':
            img = iutils.read_16bit_tif(p)
        else:
            img = imread(p) / 255.0
        return img
    def __getitem__(self, index):
        index = index + self.args.start_idx
        img_paths, exposures = self._get_input_path(index)
        img_paths.reverse()
        exposures = exposures[::-1]
        item = {}
        input_expos = []
        ref_expos = []
        # self.nframes * 2 - 1
        c_idx = self.nframes - 1
        pos = None
        for i, p in enumerate(img_paths):
            if i % 2 == 0:
                img = self.read_im(p)
                if pos is None:
                    pos = [random.randint(0, img.shape[0] - 256 - 1), random.randint(0, img.shape[1] - 256 - 1)]
                img = img[pos[0] : pos[0] + 256, pos[1] : pos[1] + 256]
                item['ldr_%d' % (i // 2)] = img
                input_expos.append(exposures[i])
            if i in [c_idx-2, c_idx-1, c_idx, c_idx+1, c_idx+2]:
                img = self.read_im(p)
                item['ref_ldr%d' % (i - c_idx+2)] = img[pos[0] : pos[0] + 256, pos[1] : pos[1] + 256]
                ref_expos.append(exposures[i])

        hdr_start, hdr_end = 2, self.nframes - 2
        for i in range(hdr_start, hdr_end):
            hdr = iutils.ldr_to_hdr(item['ldr_%d'%i], exposures[i])
            item['hdr_%d' % i] = hdr

        origin_hw = (img.shape[0], img.shape[1])
        item = self.post_process(item, img_paths)

        item['hw'] = origin_hw
        item['expos'] = torch.tensor(input_expos)
        item['ref_expos'] = ref_expos
        item['reuse_cached_data'] = False
        # print(ldr_start, ldr_end, hdr_start, hdr_end, item['reuse_cached_data'])
        return item
    def __len__(self):
        return len(self.img_list)
