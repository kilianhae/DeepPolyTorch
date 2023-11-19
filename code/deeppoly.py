from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import torch
from torchviz import make_dot

class AlgebraicBound():
    def __init__(self, ub_mult: torch.Tensor, lb_mult: torch.Tensor, ub_bias: torch.Tensor, lb_bias: torch.Tensor) -> None:
        self.ub_mult = ub_mult
        self.lb_mult = lb_mult
        self.ub_bias = ub_bias
        self.lb_bias = lb_bias

class Bound():
    def __init__(self, ub: torch.Tensor, lb: torch.Tensor) -> None:
        self.ub = ub
        self.lb = lb

# we design deeppoly to be input/output agnostic but rather just specific to the model TODO not sure if thats the right way to go
class DeepPoly(torch.nn.Module):
    def __init__(self, model: torch.nn.Sequential, true_label: int):
        super().__init__()
        verifiers = [] # type: list[Verifier]
        self.input_verifier = InputVerifier()
        verifiers.append(self.input_verifier)
        
        for module in model:
            if isinstance(module, torch.nn.Linear):
                verifiers.append(LinearVerifier(layer=module, previous=verifiers[-1]))
            elif isinstance(module, torch.nn.ReLU):
                verifiers.append(ReluVerifier(previous=verifiers[-1]))
            elif isinstance(module, torch.nn.LeakyReLU):
                if module.negative_slope < 1:
                    verifiers.append(LeakyReluVerifierFlat(negative_slope=module.negative_slope,previous=verifiers[-1]))
                else: verifiers.append(LeakyReluVerifierSteep(negative_slope=module.negative_slope,previous=verifiers[-1]))
            elif isinstance(module, torch.nn.Flatten):
                pass
            # elif isinstance(module, torch.nn.Conv2d):
            #     self.verifiers.append(Conv2DVerifier(module, self.verifiers[-1]))
            else:
                raise NotImplementedError
        
        self.output_verifier = FinalLossVerifier(verifiers[-1], true_label)
        verifiers.append(self.output_verifier)
        
        self.verifiers = torch.nn.Sequential(*verifiers)
        

    def forward(self, x: torch.Tensor, eps: float) -> Bound:
        
        #construct the input bound
        ub_in = torch.Tensor.clamp(x + eps, min=0, max=1)
        lb_in = torch.Tensor.clamp(x - eps, min=0, max=1)
        input_bound = Bound(ub=ub_in,lb=lb_in)

        ## Optimization:
        if len(list(self.verifiers.parameters())) != 0 :
            opt = torch.optim.Adam(self.verifiers.parameters(), lr=1.5)
            for i in range(0,40):
                opt.zero_grad()
                final_bound = self.verifiers.forward(input_bound)
                loss = torch.sum(- final_bound.lb)
                loss.backward()
                opt.step()

        final_bound = self.verifiers.forward(Bound(ub=ub_in,lb=lb_in))
        return final_bound


class Verifier(ABC, torch.nn.Module):

    # @property
    # @abstractmethod
    # def in_size(self) -> int:
    #     return self._in_size
    
    # @property
    # @abstractmethod
    # def out_size(self) -> int:
    #     return self._in_size
    
    @abstractmethod
    def forward(self, x: Bound) -> Bound:
        # sets the current uc and lb
        # calls the next forward method and returns the bounds of the output (the final layer)
        # calls the backward and based on that sets the lb and ub of the input
        pass

    @abstractmethod
    def backward(self, bound: AlgebraicBound) -> None:
        # uc: expects to get a tensor of tensors. Each tensor corresponds to the algebraic expression of the upper bound of a output neuron (from where the backward was started). Therefore the element t_i is the mult. const in the upper bound of the output neuron i of the current layer
        # lc: expects to get a tensor of tensors. Each tensor corresponds to the algebraic expression of the lower bound of a output neuron (from where the backward was started). Therefore the element t_i is the mult. const in the lower bound of the output neuron i of the current layer
        # transofrms this so that each vector now corresponds to the mult elelements with regards to its inputs (and depending on the sign and greater or smaller than)
        # returns the real valued bounds of the inputs
        pass


class InputVerifier(Verifier):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.bound = None # type: Optional[Bound]
        self.in_size = None # type: Optional[int]
        self.out_size = None # type: Optional[int]
        
        # ub_in = torch.Tensor.clamp(x + eps, min=0, max=1)
        # lb_in = torch.Tensor.clamp(x - eps, min=0, max=1)
        # self.bound = Bound(ub=ub_in.flatten(), lb=lb_in.flatten())
        # self.in_size = self.bound.ub.size(0)
        # self.out_size = self.bound.ub.size(0)

    def forward(self, x: Bound) -> Bound:
        # algebraic_input_bound = AlgebraicBound(torch.eye(self.out_size), torch.eye(self.out_size), torch.zeros(self.out_size), torch.zeros(self.out_size))
        # self.backward(algebraic_input_bound)
        # self.bound = Bound(lb=algebraic_input_bound.lb_bias, ub=algebraic_input_bound.ub_bias)
        self.bound = Bound(lb=x.lb.flatten(), ub=x.ub.flatten())
        return self.bound
    
    def backward(self, bound: AlgebraicBound) -> None:
        """
        Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Here we dont have to backprop to the previous layer but rather we set all multiplicates to 0 and return the numerical bounds stored in the Bias attribute of the bound.
        """
        assert self.bound is not None

        bound.ub_bias = bound.ub_bias + (torch.where(bound.ub_mult>0, bound.ub_mult, 0) * self.bound.ub).sum(dim=1) + (torch.where(bound.ub_mult<0, bound.ub_mult, 0) * self.bound.lb).sum(dim=1)
        bound.lb_bias = bound.lb_bias + (torch.where(bound.lb_mult>0, bound.lb_mult, 0) * self.bound.lb).sum(dim=1) + (torch.where(bound.lb_mult<0, bound.lb_mult, 0) * self.bound.ub).sum(dim=1)

        bound.ub_mult = torch.zeros_like(bound.ub_mult)
        bound.lb_mult = torch.zeros_like(bound.lb_mult)
        


class LinearVerifier(Verifier):
    """
    Initiliazid in the forward method of a Transformer and passed backwards until the input variables at each step changing its algebraic representation.
    """
    def __init__(self, layer: torch.nn.Linear, previous: Verifier):
        torch.nn.Module.__init__(self)
        self.previous = previous
        self.weights = layer.weight.detach() # type: torch.Tensor
        self.biases = layer.bias.detach() # type: torch.Tensor

        self.bound = None # type: Optional[Bound]
        self.in_size = self.weights.size(1) # type: int
        self.out_size = self.weights.size(0) # type: int
        

    def forward(self, x: Bound) -> Bound:
        # here first we have to compute
        lb, ub = x.lb, x.ub
        # create an identity matrix with dimensions of the output vector of the linear layer (equal to the first dim of the weight matrix)
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

        bound.ub_bias = bound.ub_bias + bound.ub_mult @ self.biases
        bound.lb_bias = bound.lb_bias + bound.lb_mult @ self.biases
        bound.ub_mult = bound.ub_mult @ self.weights
        bound.lb_mult = bound.lb_mult @ self.weights
        self.previous.backward(bound)


    
class ReluVerifier(Verifier):
    """
    Initiliazid in the forward method of a Transformer and passed backwards until the input variables at each step changing its algebraic representation.
    """
    def __init__(self, previous: Verifier):
        torch.nn.Module.__init__(self)
        self.previous = previous
        self.out_size = self.previous.out_size
        self.alpha = torch.nn.Parameter(torch.zeros(self.out_size)).requires_grad_(True)
        
    def forward(self, x: Bound) -> Bound:

        # here first we have to compute
        lb, ub = x.lb, x.ub
        # need to clamp the slope so we dont compute negative slopes
        self.slope = torch.clamp(ub/(ub-lb),min=0)
        self.ub_mult = torch.diag(torch.where(lb>0,1.0,self.slope))

        self.lb_mult = torch.where(lb>0,1.0,torch.sigmoid(self.alpha))
        self.lb_mult = torch.diag(torch.where(ub<0,0,self.lb_mult))

        self.ub_bias = torch.where(lb>0,0,(- self.slope * lb))
        self.lb_bias = torch.zeros_like(lb)

        bound = AlgebraicBound(torch.eye(ub.size(0)), torch.eye(ub.size(0)), torch.zeros(ub.size(0)), torch.zeros(ub.size(0)))
        self.backward(bound)
        self.ub = bound.ub_bias
        self.lb = bound.lb_bias
        return Bound(lb=self.lb, ub=self.ub)
    
    def backward(self, bound: AlgebraicBound) -> None:
        """
        Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Then propagates the bounds to the previous layer.
        """

        bound.ub_bias = bound.ub_bias + torch.where(bound.ub_mult>0, bound.ub_mult, 0) @ self.ub_bias + torch.where(bound.ub_mult<0, bound.ub_mult, 0) @ self.lb_bias
        bound.lb_bias = bound.lb_bias + torch.where(bound.lb_mult>0, bound.lb_mult, 0) @ self.lb_bias + torch.where(bound.lb_mult<0, bound.lb_mult, 0) @ self.ub_bias

        bound.ub_mult = torch.where(bound.ub_mult>0, bound.ub_mult, 0) @ self.ub_mult + torch.where(bound.ub_mult<0, bound.ub_mult, 0) @ self.lb_mult
        bound.lb_mult = torch.where(bound.lb_mult>0, bound.lb_mult, 0) @ self.lb_mult + torch.where(bound.lb_mult<0, bound.lb_mult, 0) @ self.ub_mult 
    
        self.previous.backward(bound)


class LeakyReluVerifierFlat(Verifier):
    """
    Initiliazid in the forward method of a Transformer and passed backwards until the input variables at each step changing its algebraic representation.
    """
    def __init__(self, negative_slope: float, previous: Verifier):
        torch.nn.Module.__init__(self)
        self.previous = previous
        self.negative_slope = negative_slope
        self.out_size = self.previous.out_size
        self.alpha = torch.nn.Parameter(torch.zeros(self.out_size)).requires_grad_(True)

    def forward(self, x: Bound) -> Bound:

        # here first we have to compute
        lb, ub = x.lb, x.ub
        # need to clamp the slope so we dont compute negative slopes

        self.slope = (ub-self.negative_slope*lb)/(ub-lb)
        self.ub_mult = torch.where(lb>0,1.0,self.slope)
        self.ub_mult = torch.diag(torch.where(ub<0,self.negative_slope,self.ub_mult))
        
        normalized_alphas = torch.sigmoid(self.alpha)*(1-self.negative_slope) + self.negative_slope
        self.lb_mult = torch.where(lb>0,1.0,normalized_alphas)
        self.lb_mult = torch.diag(torch.where(ub<0,self.negative_slope,self.lb_mult))

        self.ub_bias = torch.where(lb>0, 0, (self.negative_slope - self.slope) * lb)
        self.ub_bias = torch.where(ub<0, 0, self.ub_bias)
        self.lb_bias = torch.zeros_like(lb)

        bound = AlgebraicBound(torch.eye(ub.size(0)), torch.eye(ub.size(0)), torch.zeros(ub.size(0)), torch.zeros(ub.size(0)))
        self.backward(bound)
        self.ub = bound.ub_bias
        self.lb = bound.lb_bias
        return Bound(lb=self.lb, ub=self.ub)
    
    def backward(self, bound: AlgebraicBound) -> None:
        """
        Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Then propagates the bounds to the previous layer.
        """
        bound.ub_bias = bound.ub_bias + torch.where(bound.ub_mult>0, bound.ub_mult, 0) @ self.ub_bias + torch.where(bound.ub_mult<0, bound.ub_mult, 0) @ self.lb_bias
        bound.lb_bias = bound.lb_bias + torch.where(bound.lb_mult>0, bound.lb_mult, 0) @ self.lb_bias + torch.where(bound.lb_mult<0, bound.lb_mult, 0) @ self.ub_bias

        bound.ub_mult = torch.where(bound.ub_mult>0, bound.ub_mult, 0) @ self.ub_mult + torch.where(bound.ub_mult<0, bound.ub_mult, 0) @ self.lb_mult
        bound.lb_mult = torch.where(bound.lb_mult>0, bound.lb_mult, 0) @ self.lb_mult + torch.where(bound.lb_mult<0, bound.lb_mult, 0) @ self.ub_mult 
    
        self.previous.backward(bound)
    

class LeakyReluVerifierSteep(Verifier):
    """
    Initiliazid in the forward method of a Transformer and passed backwards until the input variables at each step changing its algebraic representation.
    """
    def __init__(self, negative_slope: float, previous: Verifier):
        torch.nn.Module.__init__(self)
        self.previous = previous
        self.negative_slope = negative_slope
        self.out_size = self.previous.out_size
        self.alpha = torch.nn.Parameter(torch.zeros(self.out_size)).requires_grad_(True)

    def forward(self, x: Bound) -> Bound:

        # here first we have to compute
        lb, ub = x.lb, x.ub
        # need to clamp the slope so we dont compute negative slopes

        self.slope = (ub-self.negative_slope*lb)/(ub-lb)
        normalized_alphas = torch.sigmoid(self.alpha)*(self.negative_slope-1) + 1.0

        self.ub_mult = torch.where(lb>0,1.0,normalized_alphas)
        self.ub_mult = torch.diag(torch.where(ub<0,self.negative_slope,self.ub_mult))
        
        
        self.lb_mult = torch.where(lb>0,1.0,self.slope)
        self.lb_mult = torch.diag(torch.where(ub<0,self.negative_slope,self.lb_mult))

        offset = - self.negative_slope * lb
        self.lb_bias = torch.where(lb>0, 0, (self.negative_slope - self.slope)*lb)
        self.lb_bias = torch.where(ub<0, 0, self.lb_bias)
        self.ub_bias = torch.zeros_like(ub)

        bound = AlgebraicBound(torch.eye(ub.size(0)), torch.eye(ub.size(0)), torch.zeros(ub.size(0)), torch.zeros(ub.size(0)))
        self.backward(bound)
        self.ub = bound.ub_bias
        self.lb = bound.lb_bias
        return Bound(lb=self.lb, ub=self.ub)
    
    def backward(self, bound: AlgebraicBound) -> None:
        """
        Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Then propagates the bounds to the previous layer.
        """

        bound.ub_bias = bound.ub_bias + torch.where(bound.ub_mult>0, bound.ub_mult, 0) @ self.ub_bias + torch.where(bound.ub_mult<0, bound.ub_mult, 0) @ self.lb_bias
        bound.lb_bias = bound.lb_bias + torch.where(bound.lb_mult>0, bound.lb_mult, 0) @ self.lb_bias + torch.where(bound.lb_mult<0, bound.lb_mult, 0) @ self.ub_bias

        bound.ub_mult = torch.where(bound.ub_mult>0, bound.ub_mult, 0) @ self.ub_mult + torch.where(bound.ub_mult<0, bound.ub_mult, 0) @ self.lb_mult
        bound.lb_mult = torch.where(bound.lb_mult>0, bound.lb_mult, 0) @ self.lb_mult + torch.where(bound.lb_mult<0, bound.lb_mult, 0) @ self.ub_mult 
    
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


class FinalLossVerifier(Verifier):
    """
    Ued as last verifier layer and gives us the loss back
    """
    def __init__(self, previous: Verifier, true_label: int):
        torch.nn.Module.__init__(self)
        self.previous = previous
        self.true_label = true_label
        self.out_size = self.previous.out_size

    def forward(self, x: Bound) -> Bound:
        # here first we have to compute
        lb, ub = x.lb, x.ub

        lb_mult = torch.zeros([torch.flatten(ub).size(0)-1, torch.flatten(ub).size(0)])
        ub_mult = torch.zeros([torch.flatten(ub).size(0)-1, torch.flatten(ub).size(0)])
        
        lb_mult[:,self.true_label] = 1
        ub_mult[:,self.true_label] = 1

        for i in range(torch.flatten(ub).size(0)):
            if i < self.true_label:
                lb_mult[i,i] = -1
                ub_mult[i,i] = -1
            elif i > self.true_label:
                lb_mult[i-1,i] = -1
                ub_mult[i-1,i] = -1

        bound = AlgebraicBound(ub_mult, lb_mult, torch.zeros(torch.flatten(ub).size(0)-1), torch.zeros(torch.flatten(ub).size(0)-1))

        self.backward(bound)
        self.ub = bound.ub_bias
        self.lb = bound.lb_bias
        return Bound(lb=bound.lb_bias, ub=bound.ub_bias)
    
    def backward(self, bound: AlgebraicBound) -> None:
        """
        Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Then propagates the bounds to the previous layer.
        """
        
        self.previous.backward(bound)



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