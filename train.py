from lut3d import *
import torch
import torch.optim as optim
from Dataloader import *
from torch.utils.data import DataLoader
import os

def lut_loss(lut):
    less = (lut[(lut < 0)]) ** 2
    upper = (lut[(lut > 1)] - 1) ** 2
    dx = lut[:, :-1, :, :] - lut[:, 1:, :, :]
    dy = lut[:, :, :-1, :] - lut[:, :, 1:, :]
    dz = lut[:, :, :, :-1] - lut[:, :, :, 1:]
    mn =  torch.relu(dx).mean() + torch.relu(dy).mean() + torch.relu(dz).mean()
    tv =  torch.mean(dx ** 2) + torch.mean(dy ** 2)  + torch.mean(dz ** 2)
    return less.sum() + upper.sum() + mn + tv
creition = torch.nn.L1Loss()

if __name__=='__main__':
    lut = Lut3D()
    optimizer = optim.Adam(lut.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    save_dir = 'saved_model'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_params = os.path.join(save_dir, "model.pth")
    
    dataset = TrainDataset()
    dataloader = DataLoader(dataset, 8, shuffle=True)

    epoch = 500
    for i in range(epoch):
        j = 0
        total_loss = 0.
        total_lut_loss = 0.
        total_l1_loss = 0.
        for imgs, targets in dataloader:
            j += 1
            imgs = torch.permute(imgs, (0,3,1,2))
            targets = torch.permute(targets, (0,3,1,2))
            print('\r' + "[Epoch: {} Batch: {:2d}/{}]".format(i+1, j, len(dataloader)), flush=True, end="")

            new_img = lut(imgs)
            lut_l = lut_loss(lut.LUT)
            l1_l = creition(new_img, targets)
            loss = lut_l + l1_l
            total_loss += loss.detach().data
            total_lut_loss += lut_l.detach().data
            total_l1_loss += l1_l.detach().data
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss /= j
        total_l1_loss /= j
        total_lut_loss /= j
        if (i+1) % 100 == 0:
            print()
            print("Epoch: {}, loss: {:4f}, lut loss {:.4f}, l1 loss {:.4f}".format(i+1, total_loss, total_lut_loss, total_l1_loss))
            torch.save(lut.state_dict(), save_params)
    torch.save(lut.state_dict(), save_params)