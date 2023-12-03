import torch
import numpy as np
from bound import Bound, AlgebraicBound
from verify.verify import Verifier, Bound, AlgebraicBound
import time
class LinearVerifier(Verifier):
    """
    A Verifier for a torch.nn.Linear layer.
    """
    def __init__(self, layer: torch.nn.Linear, previous: Verifier):
        torch.nn.Module.__init__(self)
        self.previous = previous
        self.weights = layer.weight.detach()
        self.biases = layer.bias.detach()
        self.bound = None
        self._out_size = self.weights.size(0)
        self._out_dims = (self._out_size,)
        
    def forward(self, x: Bound) -> Bound:
        # create an identity matrix with dimensions of the output vector of the linear layer (equal to the first dim of the weight matrix)
        algebraic_bound = AlgebraicBound(torch.eye(self.out_size), torch.eye(self.out_size), torch.zeros(self.out_size), torch.zeros(self.out_size))
        self.backward(algebraic_bound)
        self.bound = Bound(ub=algebraic_bound.ub_bias, lb=algebraic_bound.lb_bias)
        return self.bound
    
    def backward(self, bound: AlgebraicBound) -> None:
        bound.ub_bias = bound.ub_bias + bound.ub_mult @ self.biases
        bound.lb_bias = bound.lb_bias + bound.lb_mult @ self.biases
        bound.ub_mult = bound.ub_mult @ self.weights
        bound.lb_mult = bound.lb_mult @ self.weights
        self.previous.backward(bound)


    

## NOTE: The following code is not used in the current implementation of DeepPoly. It is kept here for future reference. 
# class FlattenVerifier(Verifier):
#     """
#     Initiliazid in the forward method of a Transformer and passed backwards until the input variables at each step changing its algebraic representation.
#     """
#     def __init__(self, previous: Optional[Verifier]):
#         super().__init__(previous=previous)

#     def forward(self: Bound) -> Bound:
#         # here first we have to compute
#         lb, ub = self.previous.lb, self.previous.ub
#         bound = AlgebraicBound(torch.eye(torch.flatten(ub).size(0)), torch.eye(torch.flatten(ub).size(0)), torch.zeros(torch.flatten(ub).size(0)), torch.zeros(torch.flatten(ub).size(0)))
#         self.backward(bound)
#         self.ub = bound.ub_bias
#         self.lb = bound.lb_bias        
#         return self.next.forward()

#     def backward(self, bound: AlgebraicBound) -> None:
#         """
#         Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
#         Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
#         Then propagates the bounds to the previous layer.
#         """
#         bound.ub_mult = torch.reshape(bound.ub_mult, tuple(dim for dim in torch.concatenate((torch.tensor(bound.ub_mult.size(0))[None],torch.tensor(self.previous.ub.size())),dim=0)))
#         bound.lb_mult = torch.reshape(bound.lb_mult, tuple(dim for dim in torch.concatenate((torch.tensor(bound.lb_mult.size(0))[None],torch.tensor(self.previous.lb.size())),dim=0)))

#         # print(bound.ub_mult.size())
#         # MULT was of size eg: 100 x 100 but we need it to be 10 x 10 x 10 x 10
#         return self.previous.backward(bound)




class Conv2DVerifier(Verifier):
    """
    Initiliazid in the forward method of a Transformer and passed backwards until the input variables at each step changing its algebraic representation.
    """
    def __init__(self, layer: torch.nn.Conv2d, previous: Verifier):
        torch.nn.Module.__init__(self)
        self.previous = previous
        # for param in layer.parameters():
        #     param.requires_grad = False
        #self.layer = torch.jit.freeze(torch.jit.script(layer))
        self.kernel_size = layer.kernel_size
        self.in_channels = layer.in_channels
        self.out_channels = layer.out_channels
        self.padding = layer.padding
        self.stride = layer.stride
        self.dilation = layer.dilation


        self.weights_unflattened = layer.weight.detach() # type: torch.Tensor # shape: (out_channels, in_channels, kernel_size[0], kernel_size[1])
        self.weights = torch.flatten(layer.weight, start_dim=1).detach() # type: torch.Tensor # shape: (out_channels, in_channels * kernel_size[0] * kernel_size[1]) I THINK!
        if layer.bias is not None:
            self.biases = layer.bias.detach() 
        else:
            self.biases = torch.zeros(self.weights.size(0)).detach() 
        self.bound = None
       
       ## we stroe the dimesnions of the unflattened inputs and outputs of the conv layer for use on the backward or forward methods
        self._out_dims = self.compute_out_dims()
        self._out_size = int(torch.tensor(self._out_dims).prod().item())

        self.mult = None
        self.bias = None
        

    ## TODO: Check if this is correct!
    def compute_out_dims(self) -> tuple[int, int, int]:
        """
        Computes the output dimensions of the conv layer given the input dimensions.
        """
        weight_dimensions = self.weights_unflattened.size()
        assert isinstance(self.padding, tuple)
        out_channels = int(weight_dimensions[0])
        out_height = int((self.previous.out_dims[1]+2*self.padding[0]-self.dilation[0]*(weight_dimensions[2]-1)-1)/self.stride[0] + 1)
        out_width = int((self.previous.out_dims[2]+2*self.padding[1]-self.dilation[1]*(weight_dimensions[3]-1)-1)/self.stride[1] + 1)
        out_dims = (out_channels, out_height, out_width)
        return out_dims
        
    def forward(self, x: Bound) -> Bound:
        # create an identity matrix with dimensions of the output vector of the flattened conv layer (NOT equal to the first dim of the weight matrix)
        if self.mult is None:
            self.mult, self.bias = self.compute_mult()
        algebraic_bound = AlgebraicBound(torch.eye(self.out_size), torch.eye(self.out_size), torch.zeros(self.out_size), torch.zeros(self.out_size))
        self.backward(algebraic_bound)
        self.bound = Bound(ub=algebraic_bound.ub_bias, lb=algebraic_bound.lb_bias)
        return self.bound
    
    def compute_mult(self) -> tuple[torch.Tensor, torch.Tensor]:
        start_time = time.time()
        padding = self.padding[0]
        stride = self.stride[0]
        kernel_size = self.kernel_size[0]
        in_channels = self.in_channels
        out_channels = self.out_channels

        assert isinstance(padding, int)

        out_channels, out_height, out_width = self.out_dims

        in_width_padded = self.previous.out_dims[2] + padding * 2
        in_height_padded = self.previous.out_dims[1] + padding * 2

        in_channel_size_padded = in_height_padded * in_width_padded
        in_size_padded = in_channel_size_padded * in_channels
        mult_matrix = torch.zeros((self.out_size, in_size_padded))

        kernel_row_length = (in_channels - 1) * in_channel_size_padded + (kernel_size - 1) * in_width_padded + kernel_size
        kernel_rows = torch.zeros((out_channels, kernel_row_length))

        for oc in range(out_channels):
            for ic in range(in_channels):
                for k in range(kernel_size):
                    start_idx = ic * in_channel_size_padded + k * in_width_padded
                    end_idx = start_idx + kernel_size
                    kernel_rows[oc, start_idx:end_idx] = self.weights_unflattened[oc, ic, k]

            for ih in range(out_height):
                for iw in range(out_width):
                    start_idx = ih * stride * in_width_padded + iw * stride
                    end_idx = start_idx + kernel_row_length
                    output_idx = oc * out_height * out_width + ih * out_width + iw
                    mult_matrix[output_idx, start_idx:end_idx] = kernel_rows[oc]

        if padding != 0:
            padding_mask = torch.zeros([in_channels, in_width_padded, in_height_padded])
            padding_mask[:, padding:-padding, padding:-padding] = torch.ones(self.previous.out_dims)
            padding_mask = padding_mask.flatten()
            mult_matrix = mult_matrix[:, torch.tensor(padding_mask) == 1]

        if self.biases is None:
            bias_vector = torch.zeros(out_width * out_height * out_channels)
        else:
            bias_vector = torch.repeat_interleave(self.biases, out_width * out_height)

        computation_time = time.time() - start_time
        print("Time to compute multiplication matrix: ", computation_time)
        return mult_matrix, bias_vector

    def backward(self, bound: AlgebraicBound) -> None:
        """
        Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the 
        current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Then propagates the bounds to the previous layer.
        """
        now = time.time()
        bound.ub_bias = bound.ub_bias + bound.ub_mult @ self.bias
        bound.lb_bias = bound.lb_bias + bound.lb_mult @ self.bias
        bound.ub_mult = bound.ub_mult @ self.mult
        bound.lb_mult = bound.lb_mult @ self.mult
        print("time to compute backward: ", time.time()-now)

        return self.previous.backward(bound)