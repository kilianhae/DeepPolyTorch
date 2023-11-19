import torch

class AlgebraicBound():
    """
    Represents the algebraic upper and lower bounds on the output vars of a layer.

    Attributes
    ----------
    ub_mult : Tensor of shape (verifier_in_forward_pass.output_size, verifier_in_backward_pass.out_size) representing the Upper Bounds of the output vars of the layer
    lb_mult : Tensor of shape (verifier_in_forward_pass.output_size, verifier_in_backward_pass.out_size) representing the Lower Bounds of the output vars of the layer
    ub_bias : Tensor of shape (verifier_in_forward_pass.output_size) representing the Upper Bounds of the output vars of the layer
    lb_bias : Tensor of shape (verifier_in_forward_pass.output_size) representing the Lower Bounds of the output vars of the layer
    """
    def __init__(self, ub_mult: torch.Tensor, lb_mult: torch.Tensor, ub_bias: torch.Tensor, lb_bias: torch.Tensor) -> None:
        self.ub_mult = ub_mult
        self.lb_mult = lb_mult
        self.ub_bias = ub_bias
        self.lb_bias = lb_bias

class Bound():
    """
    Represents the numerical bounds of a layer.

    Attributes
    ----------
    ub : Tensor of shape (Verifer.out_size) representing the Upper Bounds of the outputs of the layer
    lb : Tensor of shape (Verifer.out_size) representing the Lower Bounds of the outputs of the layer
    """
    def __init__(self, ub: torch.Tensor, lb: torch.Tensor) -> None:
        self.ub = ub
        self.lb = lb