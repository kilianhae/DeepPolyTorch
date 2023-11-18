from abc import ABC
from typing import Optional
import torch

# we design deeppoly to be input/output agnostic but rather just specific to the model TODO not sure if thats the right way to go
class DeepPoly(torch.nn.Module):
    def __init__(self, model: torch.nn.Sequential, true_label: int):
        super().__init__()
        self.verifiers = [] # type: list[Verifier]
        self.input_verifier = InputVerifier(None)
        self.output_verifier = FinalLossVerifier(None, true_label)
        self.verifiers.append(self.input_verifier)
        
        print("Model: ", model)
        for module in model:
            if isinstance(module, torch.nn.Linear):
                self.verifiers.append(LinearVerifier(layer=module, previous=self.verifiers[-1]))
            elif isinstance(module, torch.nn.ReLU):
                self.verifiers.append(ReluVerifier(previous=self.verifiers[-1]))
            elif isinstance(module, torch.nn.Flatten):
                pass
            elif isinstance(module, torch.nn.Conv2d):
                self.verifiers.append(Conv2DVerifier(module, self.verifiers[-1]))
            else:
                print(module)
                raise NotImplementedError
        
        self.output_verifier.previous = self.verifiers[-1]
        self.verifiers.append(self.output_verifier)
        print("Verifiers: ", self.verifiers)
        self.verifiers = torch.nn.Sequential(*self.verifiers)
        

    def forward(self, x: torch.Tensor, eps: float) -> (torch.Tensor, torch.Tensor):
        # this method runs the whole deeploly scheme and returns wether we are correct or not
        # x is a batch of a single sample
        # y is a tensor containing single element (the target of the sample x)
        ub_in = torch.Tensor.clamp(x + eps, min=0, max=1)
        lb_in = torch.Tensor.clamp(x - eps, 0, 1)

        final_bound = self.verifiers.forward(Bound(lb_in, ub_in))
        lb, ub = final_bound.lb, final_bound.ub
        return lb, ub
        # we need to check if the true label is within the bounds

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


class Verifier(ABC):
    def __init__(self, lb=None, ub=None, previous=None) -> None:
        self.lb = lb # type: Optional[torch.Tensor]
        self.ub = ub # type: Optional[torch.Tensor]
        self.previous = previous # type: Optional[Verifier]

    def forward(self): # not sure if float as input is true
        # sets the current uc and lb
        # calls the next forward method and returns the bounds of the output (the final layer)
        # calls the backward and based on that sets the lb and ub of the input
        pass

    def backward(self, bound: AlgebraicBound):
        # uc: expects to get a tensor of tensors. Each tensor corresponds to the algebraic expression of the upper bound of a output neuron (from where the backward was started). Therefore the element t_i is the mult. const in the upper bound of the output neuron i of the current layer
        # lc: expects to get a tensor of tensors. Each tensor corresponds to the algebraic expression of the lower bound of a output neuron (from where the backward was started). Therefore the element t_i is the mult. const in the lower bound of the output neuron i of the current layer
        # transofrms this so that each vector now corresponds to the mult elelements with regards to its inputs (and depending on the sign and greater or smaller than)
        # returns the real valued bounds of the inputs
        pass


class InputVerifier(Verifier, torch.nn.Module):
    def __init__(self, previous: Optional[Verifier]):
        torch.nn.Module.__init__(self)
        Verifier.__init__(self,previous=previous)
        
    def forward(self, x: Bound) -> Bound:
        self.lb = x.lb.flatten()
        self.ub = x.ub.flatten()
        return self.backward(AlgebraicBound(torch.eye(self.lb.size(0)), torch.eye(self.lb.size(0)), torch.zeros(self.lb.size(0)), torch.zeros(self.lb.size(0))))
    
    
    def backward(self, bound: AlgebraicBound) -> Bound:
        """
        Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Here we dont have to backprop to the previous layer but rather we set all multiplicates to 0 and return the numerical bounds stored in the Bias attribute of the bound.
        """
        print("Input Verifier Backward Pass")

        bound.ub_bias = bound.ub_bias + (torch.where(bound.ub_mult>0, bound.ub_mult, 0) * self.ub).sum(dim=1) + (torch.where(bound.ub_mult<0, bound.ub_mult, 0) * self.lb).sum(dim=1)
        bound.lb_bias = bound.lb_bias + (torch.where(bound.lb_mult>0, bound.lb_mult, 0) * self.lb).sum(dim=1) + (torch.where(bound.lb_mult<0, bound.lb_mult, 0) * self.ub).sum(dim=1)

        bound.ub_mult = torch.zeros_like(bound.ub_mult)
        bound.lb_mult = torch.zeros_like(bound.lb_mult)
        return Bound(bound.lb_bias, bound.ub_bias)


class LinearVerifier(Verifier,torch.nn.Module):
    """
    Initiliazid in the forward method of a Transformer and passed backwards until the input variables at each step changing its algebraic representation.
    """
    def __init__(self, layer: torch.nn.Linear, previous: Verifier):
        torch.nn.Module.__init__(self)
        Verifier.__init__(self,previous=previous)
        self.layer = layer # type: torch.nn.Linear

    def forward(self, x: Bound) -> Bound:
        print("Linear Layer Forward Pass")
        # here first we have to compute
        lb, ub = x.lb, x.ub
        # create an identity matrix with dimensions of the output vector of the linear layer (equal to the first dim of the weight matrix)
        bound = AlgebraicBound(torch.eye(self.layer.weight.size(0)), torch.eye(self.layer.weight.size(0)), torch.zeros(self.layer.weight.size(0)), torch.zeros(self.layer.weight.size(0)))
        self.backward(bound)
        self.ub = bound.ub_bias
        self.lb = bound.lb_bias

        self.output_dims = self.lb.size()

        return Bound(self.lb, self.ub)
    
    def backward(self, bound: AlgebraicBound) -> Bound:
        """
        Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the 
        current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Then propagates the bounds to the previous layer.
        """
        bound.ub_bias = bound.ub_bias + bound.ub_mult @ self.layer.bias
        bound.lb_bias = bound.lb_bias + bound.lb_mult @ self.layer.bias
        bound.ub_mult = bound.ub_mult @ self.layer.weight
        bound.lb_mult = bound.lb_mult @ self.layer.weight
        self.previous.backward(bound)


    
class ReluVerifier(Verifier,torch.nn.Module):
    """
    Initiliazid in the forward method of a Transformer and passed backwards until the input variables at each step changing its algebraic representation.
    """
    def __init__(self, previous: Optional[Verifier]):
        torch.nn.Module.__init__(self)
        Verifier.__init__(self,previous=previous)
        self.alpha = torch.nn.Parameter(torch.zeros(previous.layer.weight.size(0)))

    def forward(self, x: Bound) -> Bound:
        print("Relu Layer Forward Pass")

        # here first we have to compute
        lb, ub = self.previous.lb, self.previous.ub
        self.output_dims = self.previous.output_dims
        # need to clamp the slope so we dont compute negative slopes
        self.slope = torch.clamp(ub/(ub-lb),min=0)
        self.ub_mult = torch.diag(torch.where(self.previous.lb>0,1.0,self.slope))

        self.lb_mult = torch.diag(torch.where(self.previous.lb>0,1.0,0.0))
        # self.lb_mult = torch.where(self.previous.lb>0,1.0,self.alpha)
        # self.lb_mult = torch.diag(torch.where(ub<0,0,self.lb_mult))

        self.ub_bias = torch.where(self.previous.lb>0,0,(- self.slope * self.previous.lb))
        self.lb_bias = torch.zeros_like(self.previous.lb)

        bound = AlgebraicBound(torch.eye(ub.size(0)), torch.eye(ub.size(0)), torch.zeros(ub.size(0)), torch.zeros(ub.size(0)))
        self.backward(bound)
        self.ub = bound.ub_bias
        self.lb = bound.lb_bias
        print("relu lower bound:", self.lb)
        return Bound(self.lb, self.ub)
    
    def backward(self, bound: AlgebraicBound) -> Bound:
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


class FlattenVerifier(Verifier):
    """
    Initiliazid in the forward method of a Transformer and passed backwards until the input variables at each step changing its algebraic representation.
    """
    def __init__(self, previous: Optional[Verifier]):
        super().__init__(previous=previous)

    def forward(self):
        print("Flatten Layer Forward Pass")
        # here first we have to compute
        lb, ub = self.previous.lb, self.previous.ub
        bound = AlgebraicBound(torch.eye(torch.flatten(ub).size(0)), torch.eye(torch.flatten(ub).size(0)), torch.zeros(torch.flatten(ub).size(0)), torch.zeros(torch.flatten(ub).size(0)))
        self.backward(bound)
        self.ub = bound.ub_bias
        self.lb = bound.lb_bias        
        return self.next.forward()

    def backward(self, bound: AlgebraicBound):
        """
        Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Then propagates the bounds to the previous layer.
        """
        bound.ub_mult = torch.reshape(bound.ub_mult, tuple(dim for dim in torch.concatenate((torch.tensor(bound.ub_mult.size(0))[None],torch.tensor(self.previous.ub.size())),dim=0)))
        bound.lb_mult = torch.reshape(bound.lb_mult, tuple(dim for dim in torch.concatenate((torch.tensor(bound.lb_mult.size(0))[None],torch.tensor(self.previous.lb.size())),dim=0)))

        # print(bound.ub_mult.size())
        # MULT was of size eg: 100 x 100 but we need it to be 10 x 10 x 10 x 10
        self.previous.backward(bound)


class FinalLossVerifier(Verifier,torch.nn.Module):
    """
    Ued as last verifier layer and gives us the loss back
    """
    def __init__(self, previous: Optional[Verifier], true_label: int):
        torch.nn.Module.__init__(self)
        Verifier.__init__(self,previous=previous)
        self.true_label = true_label

    def forward(self, x: Bound) -> Bound:
        print("Final Layer Forward Pass")
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
        print("Lower AlgebraicBound: ", self.lb)
        print("Upper AlgebraicBound: ", self.ub)
        return Bound(bound.lb_bias, bound.ub_bias)
    
    def backward(self, bound: AlgebraicBound) -> Bound:
        """
        Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Then propagates the bounds to the previous layer.
        """
        
        self.previous.backward(bound)



class Conv2DVerifier(Verifier):
    """
    Initiliazid in the forward method of a Transformer and passed backwards until the input variables at each step changing its algebraic representation.
    """
    def __init__(self, layer: torch.nn.Conv2d, previous: Verifier, next: Optional[Verifier]):
        super().__init__(previous=previous,next=next)
        self.layer = layer # type: torch.nn.Conv2d
        self.dims = self.layer.weight.size()
        self.weights_flattened = torch.flatten(self.layer.weight, start_dim=1)

       
    def forward(self):
        print("Linear Layer Forward Pass")
        # here first we have to compute
        lb, ub = self.previous.lb, self.previous.ub

        # calculate & store output dimensions
        self.input_dim_flattend = self.previous.ub.size()
        self.output_dims = torch.floor(torch.tensor([self.dims[0],(self.previous.output_dims[1]+2*self.layer.padding[0]-self.layer.dilation[0]*(self.dims[2]-1)-1)/self.layer.stride[0] + 1, (self.previous.output_dims[2]+2*self.layer.padding[1]-self.layer.dilation[1]*(self.dims[3]-1)-1)/self.layer.stride[1] + 1])).int().tolist()
        
        # create an identity matrix with dimensions of the output vector of the flattened conv layer (NOT equal to the first dim of the weight matrix)
        bound = AlgebraicBound(torch.eye(torch.zeros(self.output_dims).flatten().size(0)), torch.eye(torch.zeros(self.output_dims).flatten().size(0)), torch.zeros(self.output_dims).flatten(), torch.zeros(self.output_dims).flatten())
        self.backward(bound)
        self.ub = bound.ub_bias
        self.lb = bound.lb_bias
        return self.next.forward()

    def backward(self, bound: AlgebraicBound):
        """
        Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the 
        current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Then propagates the bounds to the previous layer.
        """
        # for the linear layer we dont need to differentiate between lower and upper bounds as they are the same
        # print("Linear Layer Backward Pass")
        # print(bound.lb_bias)
        # print(bound.ub_bias)
        a = []
        a.append(bound.ub_mult.size(0))
        a.append(self.previous.ub.size(0))
        bound.ub_mult = torch.zeros(a)
        bound.lb_mult = torch.zeros(a)        
        self.previous.backward(bound)