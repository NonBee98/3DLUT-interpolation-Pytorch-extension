import torch
import torch.nn as nn
class Lut1D(nn.Module):
    def __init__(self, dim=64):
        super(Lut1D, self).__init__()
        self.dim = dim
        # self.LUT = torch.linspace(0, 1, dim)
        self.LUT = torch.ones(self.dim, dtype=torch.float32)
        self.LUT = nn.Parameter(self.LUT, requires_grad=True)
    
    def get_loc_val(self, loc):
        return self.LUT[loc]

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        loc_x = x * (self.dim-1)
        loc_left = torch.floor(loc_x).long()
        loc_right = loc_left + 1
        loc_right = torch.clamp(loc_right, 0, self.dim-1).long()
        dx = loc_x - loc_left
        output = (1-dx) * self.get_loc_val(loc_left) + dx * self.get_loc_val(loc_right)

        return output
    
    @staticmethod
    def lut_loss(lut):
        less = (lut[(lut < 0)]) ** 2
        upper = (lut[(lut > 1)] - 1) ** 2
        dx = lut[:-1] - lut[1:]
        mn =  torch.relu(dx).mean()
        # tv =  torch.mean(dx ** 2)
        return less.sum() + upper.sum() + mn
