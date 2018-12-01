import torch
import numpy as np
import scipy.special
from numbers import Number

class IveFunction(torch.autograd.Function):
    '''Implements the exponentially scaled Bessel function of the first kind:
       https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ive.html'''

    @staticmethod
    def forward(self, v, z):
        assert isinstance(v, Number), 'v must be a scalar'
        
        # Keep for backward pass
        self.save_for_backward(z)
        self.v = v
        z_cpu = z.data.cpu().numpy()
        
        # Call Bessel function
        if np.isclose(v, 0):
            output = scipy.special.i0e(z_cpu, dtype=z_cpu.dtype)
        elif np.isclose(v, 1):
            output = scipy.special.i1e(z_cpu, dtype=z_cpu.dtype)
        else: 
            output = scipy.special.ive(v, z_cpu, dtype=z_cpu.dtype)
        
        return torch.Tensor(output).to(z.device)

    @staticmethod
    def backward(self, grad_output):
        z = self.saved_tensors[-1]

        # So apparently this is the gradient of the Bessel function
        return None, grad_output * (ive(self.v - 1, z) - ive(self.v, z) * (self.v + z) / z)

ive = IveFunction.apply
