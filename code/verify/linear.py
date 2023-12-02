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

        
        

    ## TODO: Check if this is correct!
    def compute_out_dims(self) -> tuple[int, ...]:
        """
        Computes the output dimensions of the conv layer given the input dimensions.
        """
        weight_dimensions = self.weights_unflattened.size()
        #assert isinstance(self.layer.padding, tuple)
        out_dims = tuple(torch.floor(torch.tensor([weight_dimensions[0],(self.previous.out_dims[1]+2*self.padding[0]-self.dilation[0]*(weight_dimensions[2]-1)-1)/self.stride[0] + 1, (self.previous.out_dims[2]+2*self.padding[1]-self.dilation[1]*(weight_dimensions[3]-1)-1)/self.stride[1] + 1])).int().tolist())
        return out_dims
        
    def forward(self, x: Bound) -> Bound:
        # create an identity matrix with dimensions of the output vector of the flattened conv layer (NOT equal to the first dim of the weight matrix)
        self.mult, self.bias = self.compute_mult()
        algebraic_bound = AlgebraicBound(torch.eye(self.out_size), torch.eye(self.out_size), torch.zeros(self.out_size), torch.zeros(self.out_size))
        self.backward(algebraic_bound)
        self.bound = Bound(ub=algebraic_bound.ub_bias, lb=algebraic_bound.lb_bias)
        return self.bound
    
    def compute_mult(self) -> tuple[torch.Tensor, torch.Tensor]:

        padding=self.padding[0]
        stride=self.stride[0]
        kernel_size=self.kernel_size[0]
        in_channels=self.in_channels
        out_channels=self.out_channels

        assert isinstance(padding, int)
        # compute all dimensions


        # in_channels * in_height * in_width = in_features
        in_features = self.previous.out_size
        # in_height = int(np.sqrt(in_features / in_channels))
        # in_width = int(np.sqrt(in_features / in_channels))

        in_channels = self.previous.out_dims[0]
        in_height = self.previous.out_dims[1]
        in_width = self.previous.out_dims[2]
        out_channels = self.out_dims[0]
        out_height = self.out_dims[1]
        out_width = self.out_dims[2]

        in_width_p = in_width + padding * 2
        in_height_p = in_height + padding * 2
        # out_height = int((in_height + padding * 2 - kernel_size) / stride + 1)
        # out_width = int((in_width + padding * 2 - kernel_size) / stride + 1)
        # out_features = out_channels * out_height * out_width
        weights = self.weights_unflattened



        size_p = in_height_p * in_width_p
        in_dim = size_p * in_channels
        out_dim = out_height * out_width * out_channels
        res = torch.zeros((out_dim, in_dim))

        # build row fillers
        len_rows = (in_channels - 1) * size_p + (kernel_size - 1) * in_width_p + kernel_size
        rows = torch.zeros((out_channels, len_rows))
        channels = torch.zeros((out_channels, len_rows))

        for i_out in range(out_channels):
            for i_in in range(in_channels):
                i_p = i_in * size_p
                for k in range(kernel_size):
                    start = i_p + k * in_width_p
                    end = start + kernel_size
                    rows[i_out, start:end] = weights[i_out, i_in, k]
                    channels[i_out, start:end] = weights[i_out, i_in, k]

            for i_out_height in range(out_height):
                for i_out_width in range(out_width):
                    start = i_out_height * stride * in_width_p + i_out_width * stride
                    end = start + len_rows
                    output = i_out * out_height * out_width + i_out_height * out_width + i_out_width
                    res[output, start:end] = channels[i_out]

        # remove padding
        padding_rows = []
        for i_in in range(in_channels):
            for i_in_height in range(in_height_p):
                for i_in_width in range(in_width_p):
                    if i_in_width < padding or i_in_width >= padding + in_width:
                        padding_rows.append(i_in * size_p + i_in_height * in_width_p + i_in_width)

                if i_in_height < padding or i_in_height >= padding + in_height:
                    start = i_in * size_p + i_in_height * in_width_p
                    end = start + in_width_p
                    padding_rows = padding_rows + list(range(start, end))

        padding_rows = list(np.unique(np.array(padding_rows)))  # delete duplicates

        lc = torch.from_numpy(np.delete(res.numpy(), padding_rows, axis=1)).detach()

        if self.biases is None:
            ret_bias = torch.zeros(out_width * out_height * out_channels)
        else:
            ret_bias = torch.repeat_interleave(self.biases, out_width * out_height)

        return lc, ret_bias

    def backward(self, bound: AlgebraicBound) -> None:
        """
        Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the 
        current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Then propagates the bounds to the previous layer.
        """
        # parameters
        bound.ub_bias = bound.ub_bias + bound.ub_mult @ self.bias
        bound.lb_bias = bound.lb_bias + bound.lb_mult @ self.bias
        bound.ub_mult = bound.ub_mult @ self.mult
        bound.lb_mult = bound.lb_mult @ self.mult
        
        return self.previous.backward(bound)