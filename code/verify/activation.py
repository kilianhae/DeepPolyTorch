
import torch
from verify.verify import Verifier, Bound, AlgebraicBound

class ReluVerifier(Verifier):
    """
    A Verifier for a torch.nn.ReLU layer.
    """
    def __init__(self, previous: Verifier):
        torch.nn.Module.__init__(self)
        self.previous = previous
        self._out_size = self.previous.out_size
        self._out_dims = self.previous.out_dims
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
        bound.ub_bias = bound.ub_bias + torch.where(bound.ub_mult>0, bound.ub_mult, 0) @ self.ub_bias + torch.where(bound.ub_mult<0, bound.ub_mult, 0) @ self.lb_bias
        bound.lb_bias = bound.lb_bias + torch.where(bound.lb_mult>0, bound.lb_mult, 0) @ self.lb_bias + torch.where(bound.lb_mult<0, bound.lb_mult, 0) @ self.ub_bias

        bound.ub_mult = torch.where(bound.ub_mult>0, bound.ub_mult, 0) @ self.ub_mult + torch.where(bound.ub_mult<0, bound.ub_mult, 0) @ self.lb_mult
        bound.lb_mult = torch.where(bound.lb_mult>0, bound.lb_mult, 0) @ self.lb_mult + torch.where(bound.lb_mult<0, bound.lb_mult, 0) @ self.ub_mult 
    
        self.previous.backward(bound)


class LeakyReluVerifierFlat(Verifier):
    """
    A Verifier for a torch.nn.LeakyReLU layer with negative slope < 1.
    """
    def __init__(self, negative_slope: float, previous: Verifier):
        torch.nn.Module.__init__(self)
        self.previous = previous
        self.negative_slope = negative_slope
        self._out_size = self.previous.out_size
        self._out_dims = self.previous.out_dims
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
        bound.ub_bias = bound.ub_bias + torch.where(bound.ub_mult>0, bound.ub_mult, 0) @ self.ub_bias + torch.where(bound.ub_mult<0, bound.ub_mult, 0) @ self.lb_bias
        bound.lb_bias = bound.lb_bias + torch.where(bound.lb_mult>0, bound.lb_mult, 0) @ self.lb_bias + torch.where(bound.lb_mult<0, bound.lb_mult, 0) @ self.ub_bias

        bound.ub_mult = torch.where(bound.ub_mult>0, bound.ub_mult, 0) @ self.ub_mult + torch.where(bound.ub_mult<0, bound.ub_mult, 0) @ self.lb_mult
        bound.lb_mult = torch.where(bound.lb_mult>0, bound.lb_mult, 0) @ self.lb_mult + torch.where(bound.lb_mult<0, bound.lb_mult, 0) @ self.ub_mult 
    
        self.previous.backward(bound)
    

class LeakyReluVerifierSteep(Verifier):
    """
    A Verifier for a torch.nn.LeakyReLU layer with negative slope > 1.
    """
    def __init__(self, negative_slope: float, previous: Verifier):
        torch.nn.Module.__init__(self)
        self.previous = previous
        self.negative_slope = negative_slope
        self._out_size = self.previous.out_size
        self._out_dims = self.previous.out_dims
        self.alpha = torch.nn.Parameter(torch.zeros(self.out_size)).requires_grad_(True)

    def forward(self, x: Bound) -> Bound:

        # here first we have to compute
        lb, ub = x.lb, x.ub
        # need to clamp the slope so we dont compute negative slopes

        slope = (ub-self.negative_slope*lb)/(ub-lb)
        normalized_alphas = torch.sigmoid(self.alpha)*(self.negative_slope-1) + 1.0

        # compute the elementwise upper and lower bound expressions and store them for future backward passes
        self.ub_mult = torch.where(lb>0,1.0,normalized_alphas)
        self.ub_mult = torch.diag(torch.where(ub<0,self.negative_slope,self.ub_mult))
        self.lb_mult = torch.where(lb>0,1.0,slope)
        self.lb_mult = torch.diag(torch.where(ub<0,self.negative_slope,self.lb_mult))

        self.lb_bias = torch.where(lb>0, 0, (self.negative_slope - slope)*lb)
        self.lb_bias = torch.where(ub<0, 0, self.lb_bias)
        self.ub_bias = torch.zeros_like(ub)

        bound = AlgebraicBound(torch.eye(ub.size(0)), torch.eye(ub.size(0)), torch.zeros(ub.size(0)), torch.zeros(ub.size(0)))
        self.backward(bound)
        self.ub = bound.ub_bias
        self.lb = bound.lb_bias
        return Bound(lb=self.lb, ub=self.ub)
    
    def backward(self, bound: AlgebraicBound) -> None:
        bound.ub_bias = bound.ub_bias + torch.where(bound.ub_mult>0, bound.ub_mult, 0) @ self.ub_bias + torch.where(bound.ub_mult<0, bound.ub_mult, 0) @ self.lb_bias
        bound.lb_bias = bound.lb_bias + torch.where(bound.lb_mult>0, bound.lb_mult, 0) @ self.lb_bias + torch.where(bound.lb_mult<0, bound.lb_mult, 0) @ self.ub_bias

        bound.ub_mult = torch.where(bound.ub_mult>0, bound.ub_mult, 0) @ self.ub_mult + torch.where(bound.ub_mult<0, bound.ub_mult, 0) @ self.lb_mult
        bound.lb_mult = torch.where(bound.lb_mult>0, bound.lb_mult, 0) @ self.lb_mult + torch.where(bound.lb_mult<0, bound.lb_mult, 0) @ self.ub_mult 
    
        self.previous.backward(bound)
