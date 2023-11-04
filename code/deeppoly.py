from abc import ABC
from typing import Optional
import torch

# we design deeppoly to be input/output agnostic but rather just specific to the model TODO not sure if thats the right way to go
class DeepPoly():
    def __init__(self, model: torch.nn.Sequential, true_label: int):
        self.verifiers = [] # type: list[Verifier]
        self.input_verifier = InputVerifier(None, None)
        self.verifiers.append(self.input_verifier)

        for module in model:
            print(isinstance(module, torch.nn.Linear))
            if isinstance(module, torch.nn.Linear):
                self.verifiers.append(LinearVerifier(module, self.verifiers[-1], None))
            elif isinstance(module, torch.nn.ReLU):
                self.verifiers.append(ReluVerifier(self.verifiers[-1], None))
            elif isinstance(module, torch.nn.Flatten):
                self.verifiers.append(FlattenVerifier(self.verifiers[-1], None))
            else:
                print(module)
                raise NotImplementedError
            if len(self.verifiers) > 1:
                self.verifiers[-2].next = self.verifiers[-1]
        
        self.input_verifier.next = self.verifiers[1]

    def forward(self, x: torch.Tensor, eps: float):
        # this method runs the whole deeploly scheme and returns wether we are correct or not
        # x is a batch of a single sample
        # y is a tensor containing single element (the target of the sample x)
        ub_in = torch.Tensor.clamp(x + eps, min=0, max=1)
        lb_in = torch.Tensor.clamp(x - eps, 0, 1)

        self.input_verifier.set_init_box(lb_in, ub_in)
        lb, ub = self.input_verifier.forward()
        return lb, ub
        # we need to check if the true label is within the bounds

class Bound():
    def __init__(self, ub_mult: torch.Tensor, lb_mult: torch.Tensor, ub_bias: torch.Tensor, lb_bias: torch.Tensor) -> None:
        self.ub_mult = ub_mult
        self.lb_mult = lb_mult
        self.ub_bias = ub_bias
        self.lb_bias = lb_bias

class Verifier(ABC):
    def __init__(self, lb=None, ub=None, previous=None, next=None) -> None:
        self.lb = lb # type: Optional[torch.Tensor]
        self.ub = ub # type: Optional[torch.Tensor]
        self.previous = previous # type: Optional[Verifier]
        self.next = next # type: Optional[Verifier]

    def forward(self): # not sure if float as input is true
        # sets the current uc and lb
        # calls the next forward method and returns the bounds of the output (the final layer)
        # calls the backward and based on that sets the lb and ub of the input
        pass

    def backward(self, bound: Bound):
        # uc: expects to get a tensor of tensors. Each tensor corresponds to the algebraic expression of the upper bound of a output neuron (from where the backward was started). Therefore the element t_i is the mult. const in the upper bound of the output neuron i of the current layer
        # lc: expects to get a tensor of tensors. Each tensor corresponds to the algebraic expression of the lower bound of a output neuron (from where the backward was started). Therefore the element t_i is the mult. const in the lower bound of the output neuron i of the current layer
        # transofrms this so that each vector now corresponds to the mult elelements with regards to its inputs (and depending on the sign and greater or smaller than)
        # returns the real valued bounds of the inputs
        pass

class InputVerifier(Verifier):
    def __init__(self, previous: Optional[Verifier], next: Optional[Verifier]):
        super().__init__(previous=previous,next=next)

    def set_init_box(self, lb: torch.Tensor, ub: torch.Tensor):
        self.lb = lb
        self.ub = ub

    def forward(self):
        return self.next.forward()
    
    def backward(self, bound: Bound):
        """
        Input is a Bound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Here we dont have to backprop to the previous layer but rather we set all multiplicates to 0 and return the numerical bounds stored in the Bias attribute of the bound.
        """
        bound.ub_bias = bound.ub_bias + torch.where(bound.ub_mult>0, bound.ub_mult*self.ub, bound.ub_mult*self.lb)
        bound.lb_bias = bound.lb_bias + torch.where(bound.lb_mult>0, bound.lb_mult*self.lb, bound.lb_mult*self.ub)
        bound.ub_mult = torch.zeros_like(bound.ub_mult)
        bound.lb_mult = torch.zeros_like(bound.ub_mult)


class LinearVerifier(Verifier):
    """
    Initiliazid in the forward method of a Transformer and passed backwards until the input variables at each step changing its algebraic representation.
    """
    def __init__(self, layer: torch.nn.Linear, previous: Verifier, next: Optional[Verifier]):
        super().__init__(previous=previous,next=next)
        self.layer = layer # type: torch.nn.Linear

    def forward(self):
        # here first we have to compute
        lb, ub = self.previous.lb, self.previous.ub
        bound = Bound(torch.eye(self.layer.weight.size(0)), torch.eye(self.layer.weight.size(0)), torch.zeros(ub.size(0)), torch.zeros(ub.size(0)))
        #bound = Bound(torch.tensor(self.layer.weight), torch.tensor(self.layer.weight), torch.tensor(self.layer.bias), torch.tensor(self.layer.bias))
        self.backward(bound)
        self.ub = bound.ub_bias
        self.lb = bound.lb_bias
        return self.next.forward()
    
    def backward(self, bound: Bound):
        """
        Input is a Bound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the 
        current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Then propagates the bounds to the previous layer.
        """

        bound.ub_bias = bound.ub_mult @ self.layer.bias
        bound.lb_bias = bound.lb_mult @ self.layer.bias
        bound.ub_mult = bound.ub_mult @ self.layer.weight
        bound.lb_mult = bound.lb_mult @ self.layer.weight

        self.previous.backward(bound)

class ReluVerifier(Verifier):
    """
    Initiliazid in the forward method of a Transformer and passed backwards until the input variables at each step changing its algebraic representation.
    """
    def __init__(self, previous: Optional[Verifier], next: Optional[Verifier]):
        super().__init__(previous=previous,next=next)

    def forward(self):
        # here first we have to compute
        lb, ub = self.previous.lb, self.previous.ub
        self.slope = ub/(ub-lb)
        bound = Bound(torch.eye(ub.size(0)), torch.eye(ub.size(0)), torch.zeros(ub.size(0)), torch.zeros(ub.size(0)))
        self.backward(bound)
        self.ub = bound.ub_bias
        self.lb = bound.lb_bias
        return self.next.forward()
    
    def backward(self, bound: Bound):
        """
        Input is a Bound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Then propagates the bounds to the previous layer.
        """
        bound.ub_bias = - self.slope * self.previous.ub
        bound.lb_bias = 0 * bound.lb_bias
        bound.ub_mult = self.slope * bound.ub_mult
        bound.lb_mult = 0 * bound.lb_bias

        self.previous.backward(bound)

class FlattenVerifier(Verifier):
    """
    Initiliazid in the forward method of a Transformer and passed backwards until the input variables at each step changing its algebraic representation.
    """
    def __init__(self, previous: Optional[Verifier], next: Optional[Verifier]):
        super().__init__(previous=previous,next=next)

    def forward(self):
        # here first we have to compute
        lb, ub = self.previous.lb, self.previous.ub
        bound = Bound(torch.eye(torch.flatten(ub).size(0)), torch.eye(torch.flatten(ub).size(0)), torch.zeros(torch.flatten(ub).size(0)), torch.zeros(torch.flatten(ub).size(0)))
        self.backward(bound)
        self.ub = bound.ub_bias
        self.lb = bound.lb_bias
        return self.next.forward()
    
    def backward(self, bound: Bound):
        """
        Input is a Bound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Then propagates the bounds to the previous layer.
        """

        bound.ub_bias = torch.reshape(bound.ub_bias, tuple(dim for dim in self.previous.ub.size()))
        bound.lb_bias = torch.reshape(bound.lb_bias, tuple(dim for dim in self.previous.ub.size()))
        # bias was of size eg: 100 x 1 but now we need it to be: 10 x 10 x 1
        # previous ub is of size 10 x 10 x 1
        print(bound.ub_mult.size())
        bound.ub_mult = torch.reshape(bound.ub_mult, tuple(dim for dim in torch.concatenate((torch.tensor(self.previous.ub.size()),torch.tensor(self.previous.ub.size())),dim=0)))
        bound.lb_mult = torch.reshape(bound.lb_mult, tuple(dim for dim in torch.concatenate((torch.tensor(self.previous.ub.size()),torch.tensor(self.previous.ub.size())),dim=0)))
        # MULT was of size eg: 100 x 100 but we need it to be 10 x 10 x 10 x 10
        self.previous.backward(bound)