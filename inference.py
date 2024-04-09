import argparse
import torch
from torch import nn
import yaml
from model.model_helper import ModelBuilder
from dataset.semi import SemiDataset
from torch.utils.data import DataLoader
import numpy as np
import os
from dataset.transform import *
from PIL import Image
from skimage.transform import resize
parser = argparse.ArgumentParser(description='Inference script for the model')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--model-path', type=str, required=True)
parser.add_argument('--data-path', type=str, required=True)
parser.add_argument('--output-path', type=str, required=True)

def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    checkpoint = torch.load(args.model_path, map_location=torch.device('cuda'))

    model = ModelBuilder(cfg['model'])
    model = nn.DataParallel(model)
    model.cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    
    img = Image.open("/kaggle/input/pascal-voc-2012/VOC2012/JPEGImages/2007_003431.jpg").convert('RGB')
    img = img.resize((500, 332))
    img = normalize(img)
    img = img.unsqueeze(0)
if __name__ == '__main__':
    main()