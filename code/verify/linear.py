import torch

from bound import Bound, AlgebraicBound
from verify.verify import Verifier, Bound, AlgebraicBound

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
        self.layer = torch.jit.freeze(torch.jit.script(layer))
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
        out_dims = tuple(torch.floor(torch.tensor([weight_dimensions[0],(self.previous.out_dims[1]+2*self.padding[0]-self.dilation[0]*(weight_dimensions[2]-1)-1)/self.stride[0] + 1, (self.previous.out_dims[2]+2*self.padding[1]-self.dilation[1]*(weight_dimensions[3]-1)-1)/self.stride[1] + 1])).tolist())
        return out_dims
        
    def forward(self, x: Bound) -> Bound:
        # create an identity matrix with dimensions of the output vector of the flattened conv layer (NOT equal to the first dim of the weight matrix)
        algebraic_bound = AlgebraicBound(torch.eye(self.out_size), torch.eye(self.out_size), torch.zeros(self.out_size), torch.zeros(self.out_size))
        self.backward(algebraic_bound)
        self.bound = Bound(ub=algebraic_bound.ub_bias, lb=algebraic_bound.lb_bias)
        return self.bound

    def backward(self, bound: AlgebraicBound) -> None:
        """
        Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the 
        current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Then propagates the bounds to the previous layer.
        """
        ub_mult = torch.zeros([bound.ub_mult.size(0), self.previous.out_size])
        lb_mult = torch.zeros([bound.lb_mult.size(0), self.previous.out_size])
        bound.ub_mult=ub_mult
        bound.lb_mult=lb_mult
        self.previous.backward(bound)



        