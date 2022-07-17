from lut3d import *
import numpy as np
import torch
import cv2
import os
import glob
from tqdm import tqdm

def inference(lut, img_file):
    basenmae = os.path.basename(img_file)
    save_dir = 'test_output'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.
    img = np.expand_dims(img, 0)
    img = torch.tensor(img)
    img = torch.permute(img, (0,3,1,2))
    
    with torch.no_grad():
        new_img = lut(img)
        imgS = new_img.cpu().detach()
        imgS = torch.squeeze(imgS)
        imgS = torch.permute(imgS, (1,2,0)).numpy()
        imgS *= 255
        imgS = imgS.astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, basenmae), imgS)
        

if __name__=='__main__':
    lut = Lut3D()
    save_dir = 'saved_model'
    save_params = os.path.join(save_dir, "model.pth")
    lut.load_state_dict(torch.load(save_params))
    test_img_dir = 'test_imgs'
    file_list = glob.glob(os.path.join(test_img_dir, "*"))
    file_list = list(filter(lambda x: x.endswith("jpg") or x.endswith("png") or x.endswith("jpeg"), file_list))
    for file in tqdm(file_list):
        inference(lut, file)
