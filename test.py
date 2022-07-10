import colour
from lut3d import *
import numpy as np
import cv2
import torch
from colour.algebra.interpolation import table_interpolation, table_interpolation_tetrahedral

if __name__=='__main__':
    lutRaw = colour.read_LUT("./35_Free_LUTs/Korben 214.CUBE")
    lut = lutRaw.table.astype(np.float32)
    lut = torch.tensor(lut)
    lut = torch.permute(lut, (3, 0, 1, 2))
    img = cv2.imread('./0003.jpg')
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
    cv2.imwrite("test.jpg", new_img)

    # new_img = table_interpolation_tetrahedral(imgRaw, lutRaw.table)
    # new_img *= 255
    # new_img = new_img.astype(np.uint8)
    # new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("test2.jpg", new_img)