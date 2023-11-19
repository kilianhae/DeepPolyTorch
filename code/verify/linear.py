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




# class Conv2DVerifier(Verifier):
#     """
#     Initiliazid in the forward method of a Transformer and passed backwards until the input variables at each step changing its algebraic representation.
#     """
#     def __init__(self, layer: torch.nn.Conv2d, previous: Verifier, next: Optional[Verifier]):
#         super().__init__(previous=previous,next=next)
#         self.layer = layer # type: torch.nn.Conv2d
#         self.dims = self.weights.size()
#         self.weights_flattened = torch.flatten(layer.weight, start_dim=1)

       
#     def forward(self):
#         print("Linear Layer Forward Pass")
#         # here first we have to compute
#         lb, ub = self.previous.lb, self.previous.ub

#         # calculate & store output dimensions
#         self.input_dim_flattend = self.previous.ub.size()
#         self.output_dims = torch.floor(torch.tensor([self.dims[0],(self.previous.output_dims[1]+2*self.layer.padding[0]-self.layer.dilation[0]*(self.dims[2]-1)-1)/self.layer.stride[0] + 1, (self.previous.output_dims[2]+2*self.layer.padding[1]-self.layer.dilation[1]*(self.dims[3]-1)-1)/self.layer.stride[1] + 1])).int().tolist()
        
#         # create an identity matrix with dimensions of the output vector of the flattened conv layer (NOT equal to the first dim of the weight matrix)
#         bound = AlgebraicBound(torch.eye(torch.zeros(self.output_dims).flatten().size(0)), torch.eye(torch.zeros(self.output_dims).flatten().size(0)), torch.zeros(self.output_dims).flatten(), torch.zeros(self.output_dims).flatten())
#         self.backward(bound)
#         self.ub = bound.ub_bias
#         self.lb = bound.lb_bias
#         return self.next.forward()

#     def backward(self, bound: AlgebraicBound):
#         """
#         Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the 
#         current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
#         Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
#         Then propagates the bounds to the previous layer.
#         """
#         # for the linear layer we dont need to differentiate between lower and upper bounds as they are the same
#         # print("Linear Layer Backward Pass")
#         # print(bound.lb_bias)
#         # print(bound.ub_bias)
#         a = []
#         a.append(bound.ub_mult.size(0))
#         a.append(self.previous.ub.size(0))
#         bound.ub_mult = torch.zeros(a)
#         bound.lb_mult = torch.zeros(a)        
#         self.previous.backward(bound)