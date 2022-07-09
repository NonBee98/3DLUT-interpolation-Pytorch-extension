import colour
from lut3d import *
import numpy as np
import cv2
import torch

if __name__=='__main__':
    lutRaw = colour.read_LUT("./35_Free_LUTs/Arabica 12.CUBE")
    lut = lutRaw.table.astype(np.float32)
    lut = torch.tensor(lut)
    lut = torch.permute(lut, (3, 0, 1, 2))
    img = cv2.imread('./0002-01.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    imgRaw = img/255.
    img = np.expand_dims(imgRaw, 0)
    img = torch.tensor(img)
    img = torch.permute(img, (0,3,1,2))
    tri_interp = TrilinearInterpolation()

    _, new_img = tri_interp(lut, img)
    new_img = new_img.cpu().detach()
    new_img = torch.squeeze(new_img)
    new_img = torch.permute(new_img, (1,2,0)).numpy()
    new_img *= 255
    new_img = new_img.astype(np.uint8)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test.jpg", new_img)

    new_img = lutRaw.apply(imgRaw)
    new_img *= 255
    new_img = new_img.astype(np.uint8)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test2.jpg", new_img)