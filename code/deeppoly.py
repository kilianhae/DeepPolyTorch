from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from typing import Optional
import torch
from torchviz import make_dot

from bound import Bound, AlgebraicBound
from verify.verify import Verifier, FinalLossVerifier, InputVerifier
from verify.linear import LinearVerifier, Conv2DVerifier
from verify.activation import ReluVerifier, LeakyReluVerifierFlat, LeakyReluVerifierSteep

class DeepPoly(torch.nn.Module):
    """
    An object that represents a DeepPoly Verifier for a given network and a given input.
    It is initialized with a model and a true label and then can be called with an input 
    and an epsilon to compute the bounds and to optimize on ReLu parameters to get tight and sound bounds.
    
    Attributes
    ----------
    model : torch.nn.Sequential representing the model to be verified
    true_label : int representing the true label of the input
    verifiers : torch.nn.Sequential representing the verifiers for each layer of the model
    input_verifier : InputVerifier representing the verifier for the input layer
    """
    def __init__(self, model: torch.nn.Sequential, input: torch.Tensor, true_label: int):
        super().__init__()
        verifiers = [] # type: list[Verifier]
        self.input_verifier = InputVerifier(input.size())
        verifiers.append(self.input_verifier)
        
        for module in model:
            if isinstance(module, torch.nn.Linear):
                verifiers.append(LinearVerifier(layer=module, previous=verifiers[-1]))
            elif isinstance(module, torch.nn.Conv2d):
                verifiers.append(Conv2DVerifier(layer=module, previous=verifiers[-1]))
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
        
        output_verifier = FinalLossVerifier(verifiers[-1], true_label)
        verifiers.append(output_verifier)
        
        self.verifiers = torch.nn.Sequential(*verifiers)
        

    def forward(self, x: torch.Tensor, eps: float) -> Bound:
        # construct the input bound
        ub_in = torch.Tensor.clamp(x + eps, min=0, max=1)
        lb_in = torch.Tensor.clamp(x - eps, min=0, max=1)
        input_bound = Bound(ub=ub_in,lb=lb_in)

        ## Optimization:
        lowest=-1
        if len(list(self.verifiers.parameters())) != 0 :
            opt = torch.optim.Adam(self.verifiers.parameters(), lr=1.5)
            for i in range(0,10):
                opt.zero_grad()
                final_bound = self.verifiers.forward(input_bound)
                if lowest >= 0:
                    print("stopped at iteration: ", i)
                    break
                lowest = torch.min(final_bound.lb)
                loss = torch.sum(- final_bound.lb)
                loss.backward()
                opt.step()

        final_bound = self.verifiers.forward(Bound(ub=ub_in,lb=lb_in))
        return final_bound


