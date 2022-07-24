from lut3d import *
import numpy as np
import torch
import cv2
import os
from lut1d import *
import colour

class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.lut1d = Lut1D(64)
        self.lut3d = Lut3D(17)
    
    def forward(self, x):
        x = self.lut1d(x)
        x = torch.clamp(x, 0, 1)
        x = self.lut3d(x)
        return x

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
    model = Mymodel()
    save_dir = 'saved_model'
    save_params = os.path.join(save_dir, "model.pth")
    model.load_state_dict(torch.load(save_params))
    
    lut = model.lut3d.LUT.data.numpy()
    lut = np.transpose(lut, (3,2,1,0))
    print(lut.shape)
    lutForsave = colour.LUT3D(lut)
    colour.write_LUT(lutForsave, "testlut.cube")