from ast import main
import torch
import cv2
import numpy as np
from lut3d import *

if __name__=='__main__':
    img = cv2.imread('./lambo.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)
    # img = np.repeat(img, 3, axis=0)

    img = torch.tensor(img / 255.).float()
    img = torch.permute(img, (0, 3, 1, 2))
    lut = Lut3D()
    new_img = lut(img)
    new_img = new_img.cpu().detach()
    new_img = torch.squeeze(new_img)
    new_img = torch.permute(new_img, (1,2,0)).numpy()
    new_img *= 255
    new_img = new_img.astype(np.uint8)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test.jpg", new_img)