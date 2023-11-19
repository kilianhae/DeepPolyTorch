
import torch
from abc import ABC, abstractmethod
from typing import Optional
from bound import Bound, AlgebraicBound

class Verifier(ABC, torch.nn.Module):
    @property
    def out_size(self) -> int:
        """
        Must be set for every Verifier (for the input verifier this is only set after the forward pass)

        Returns:
            The number of output neurons of the current layer.
        """
        assert isinstance(self._out_size, int)
        return self._out_size
    
    @abstractmethod
    def forward(self, x: Bound) -> Bound:
        """
        Does a backward pass from the current layer to the input layer and returns the 
        numerical upper and lower bounds as a Bound(ub,lb) object.
        Its output will be used as input for the next layer automatically by the DeepPoly class.

        Args:
            x: Bound(ub,lb) the numerical bounds of the previous layer
        
        Returns:
            Bound(ub,lb) the numerical bounds of the current layer
        """
        pass

    @abstractmethod
    def backward(self, bound: AlgebraicBound) -> None:
        """
        Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the current layer. 
        Recomputes the parameters of the AlgebraicBound in-place and if the Verifier is not an InputVerifier,
        then calls self.previous.backward!

        Args:
            bound: AlgebraicBound(ub_mult, lb_mult, ub_bias, lb_bias)
                ub_mult: Tensor: verifier_in_forward_pass.output_size x verifer.output_size
                ub_bias: Tensor: verifier_in_forward_pass.output_size   
        
        Returns:
            None
        """
        
        pass


class InputVerifier(Verifier):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.bound = None # type: Optional[Bound]
        self._out_size = None # type: Optional[int]
        self.previous = None # type: Optional[Verifier]
        
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
        

class FinalLossVerifier(Verifier):
    """
    Ued as last verifier layer and gives us the loss back
    """
    def __init__(self, previous: Verifier, true_label: int):
        torch.nn.Module.__init__(self)
        self.previous = previous
        self.true_label = true_label
        self._out_size = self.previous.out_size

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
        self.previous.backward(bound)
