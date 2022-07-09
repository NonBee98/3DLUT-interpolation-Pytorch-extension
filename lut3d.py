import torch
import torch.nn as nn
import trilinear
import tetrahedral

class Lut3D(nn.Module):
    def __init__(self, dim=17):
        super(Lut3D, self).__init__()

        self.LUT = torch.ones((3,dim,dim,dim), dtype=torch.float)
        self.LUT = nn.Parameter(self.LUT, requires_grad=True)
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)

        return output

class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut: torch.Tensor, x: torch.Tensor):
        output = x.new(x.size()).contiguous()
        dim = lut.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim-1)
        batch = x.size(0)
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)
        assert C == 3, "Can only interpolate 3D images!"
        
        trilinear.forward(lut.contiguous(), 
                            x.contiguous(), 
                            output,
                            dim, 
                            shift, 
                            binsize, 
                            W, 
                            H, 
                            batch)

        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]
        
        ctx.save_for_backward(*variables)
        
        return lut, output
    
    @staticmethod
    def backward(ctx, lut_grad: torch.Tensor, x_grad: torch.Tensor):
        
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])
            
        assert 1 == trilinear.backward(x.contiguous(), 
                                       x_grad.contiguous(), 
                                       lut_grad.contiguous(),
                                       dim, 
                                       shift, 
                                       binsize, 
                                       W, 
                                       H, 
                                       batch)
        return lut_grad, x_grad


class TrilinearInterpolation(torch.nn.Module):
    def __init__(self):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
        return TrilinearInterpolationFunction.apply(lut, x)



class TetrahedralInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut: torch.Tensor, x: torch.Tensor):
        output = x.new(x.size()).contiguous()
        dim = lut.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim-1)
        batch = x.size(0)
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)
        assert C == 3, "Can only interpolate 3D images!"
        
        tetrahedral.forward(lut.contiguous(), 
                            x.contiguous(), 
                            output,
                            dim, 
                            shift, 
                            binsize, 
                            W, 
                            H, 
                            batch)

        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]
        
        ctx.save_for_backward(*variables)
        
        return lut, output
    
    @staticmethod
    def backward(ctx, lut_grad: torch.Tensor, x_grad: torch.Tensor):
        
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])
            
        assert 1 == tetrahedral.backward(x.contiguous(), 
                                       x_grad.contiguous(), 
                                       lut_grad.contiguous(),
                                       dim, 
                                       shift, 
                                       binsize, 
                                       W, 
                                       H, 
                                       batch)
        return lut_grad, x_grad


class TetrahedralInterpolation(torch.nn.Module):
    def __init__(self):
        super(TetrahedralInterpolation, self).__init__()

    def forward(self, lut, x):
        return TetrahedralInterpolationFunction.apply(lut, x)
