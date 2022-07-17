import colour
from lut3d import *
import numpy as np
import cv2
import torch
import os

if __name__=='__main__':
    img_dir = './test_imgs'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_file = "./umbrellaL.png"
    basename = os.path.basename(img_file).split('.')[0]
    lutRaw = colour.read_LUT("./35_Free_LUTs/Ava 614.CUBE")
    lut = lutRaw.table.astype(np.float32)
    lut = torch.tensor(lut)
    lut = torch.permute(lut, (3, 0, 1, 2))
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    imgRaw = img/255.
    img = np.expand_dims(imgRaw, 0)
    img = torch.tensor(img)
    img = torch.permute(img, (0,3,1,2))
    interp = TrilinearInterpolation()

    _, new_img = interp(lut, img)

    new_img = new_img.cpu().detach()
    new_img = torch.squeeze(new_img)
    new_img = torch.permute(new_img, (1,2,0)).numpy()
    new_img *= 255
    new_img = new_img.astype(np.uint8)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("{}_out.jpg".format(basename), new_img)