import colour
from lut3d import *
import numpy as np
import cv2
import torch
import os
import glob
from tqdm import tqdm

if __name__=='__main__':
    img_dir = './train_img'
    save_dir = './train_img_np'
    input_dir = os.path.join(save_dir, "input")
    output_dir = os.path.join(save_dir, "target")
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img_list = glob.glob(os.path.join(img_dir, "*.jpg"))
    lutRaw = colour.read_LUT("./35_Free_LUTs/Chemical 168.CUBE")
    lut = lutRaw.table.astype(np.float32)
    lut = torch.tensor(lut)
    lut = torch.permute(lut, (3, 0, 1, 2))
    interp = TrilinearInterpolation()
    for img_file in tqdm(img_list):
        basename = os.path.basename(img_file).split('.')[0]
    
        img = cv2.imread(img_file)
        img = cv2.resize(img, (512,512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        imgRaw = img/img.max()
        img = np.expand_dims(imgRaw, 0)
        img = torch.tensor(img)
        img = torch.permute(img, (0,3,1,2))

        _, new_img = interp(lut, img)

        new_img = new_img.cpu().detach()
        new_img = torch.squeeze(new_img)
        new_img = torch.permute(new_img, (1,2,0)).numpy()
        new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
        
        input_path = os.path.join(input_dir, basename)
        target_path = os.path.join(output_dir, basename)
        np.save(input_path, imgRaw)
        np.save(target_path, new_img)