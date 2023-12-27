import torch
from abc import ABC, abstractmethod
from typing import Optional
from bound import Bound, AlgebraicBound


class Verifier(ABC, torch.nn.Module):
    """A Abstract Class that represents a Verifier for a given layer of a network."""

    @property
    def out_size(self) -> int:
        """
        Must be set for every Verifier (for the input verifier this is only set after the forward pass)

        Returns:
            The number of output neurons of the current layer.
        """
        assert isinstance(self._out_size, int)
        return self._out_size

    @property
    def out_dims(self) -> torch.Size:
        """
        Must be set for every Verifier (for the input verifier this is only set after the forward pass)

        Returns:
            The number of output neurons of the current layer.
        """
        assert isinstance(self._out_dims, torch.Size)
        return self._out_dims

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
        Recomputes the parameters of the AlgebraicBound in-place and if the Verifier is not an InputVerifier, calls self.previous.backward!

        Args:
            bound: AlgebraicBound(ub_mult, lb_mult, ub_bias, lb_bias)
                ub_mult: Tensor: verifier_in_forward_pass.output_size x verifer.output_size
                ub_bias: Tensor: verifier_in_forward_pass.output_size

        Returns:
            None
        """

        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset all trainable torch parameters of the verifier to their init values (or if randomized then init the same way).

        Args:
            None

        Returns:
            None
        """

        pass


class InputVerifier(Verifier):
    def __init__(self, input_dims: torch.Size | int):
        torch.nn.Module.__init__(self)
        self.bound = None  # type: Optional[Bound]
        self._out_size = None  # type: Optional[int]
        self._out_dims = input_dims  # type: torch.Size | int
        self.previous = None  # type: Optional[Verifier]

    def forward(self, x: Bound) -> Bound:
        """Initialize the numerical bounds of this layers output variables (by backwarding it through all previous layers) and returns them."""
        self.bound = Bound(lb=x.lb.flatten(), ub=x.ub.flatten())
        self._out_size = self.bound.ub.size(0)
        self._out_dims = x.lb.size()
        return self.bound

    def backward(self, bound: AlgebraicBound) -> None:
        """
        Recompute the parameters of the AlgebraicBound in-place to fully numerical bounds in the Bias.

        Args:
            bound: AlgebraicBound(ub_mult, lb_mult, ub_bias, lb_bias)
                ub_mult: Tensor: verifier_in_forward_pass.output_size x verifer.output_size
                ub_bias: Tensor: verifier_in_forward_pass.output_size

        Returns:
            None
        """
        assert self.bound is not None

        bound.ub_bias = (
            bound.ub_bias
            + (torch.where(bound.ub_mult > 0, bound.ub_mult, 0) * self.bound.ub).sum(
                dim=1
            )
            + (torch.where(bound.ub_mult < 0, bound.ub_mult, 0) * self.bound.lb).sum(
                dim=1
            )
        )
        bound.lb_bias = (
            bound.lb_bias
            + (torch.where(bound.lb_mult > 0, bound.lb_mult, 0) * self.bound.lb).sum(
                dim=1
            )
            + (torch.where(bound.lb_mult < 0, bound.lb_mult, 0) * self.bound.ub).sum(
                dim=1
            )
        )

        bound.ub_mult = torch.zeros_like(bound.ub_mult)
        bound.lb_mult = torch.zeros_like(bound.lb_mult)

    def reset(self):
        """Reset the Parameters of the verifier to their initial values."""
        print("No params to reset")


class FinalLossVerifier(Verifier):
    """
    Ued as last verifier layer and gives us the loss back
    """

    def __init__(self, previous: Verifier, true_label: int):
        torch.nn.Module.__init__(self)
        self.previous = previous
        self.true_label = true_label
        self._out_dims = self.previous.out_dims
        self._out_size = self.previous.out_size

    def forward(self, x: Bound) -> Bound:
        """Calculate the numerical bounds of this layers output variables (by backwarding it through all previous layers) and returns them.

        With 'outputs' here we mean the difference between the max logprob and the logprob of the true class.
        """
        _, ub = x.lb, x.ub

        lb_mult = torch.zeros(
            [torch.flatten(ub).size(0) - 1, torch.flatten(ub).size(0)]
        )
        ub_mult = torch.zeros(
            [torch.flatten(ub).size(0) - 1, torch.flatten(ub).size(0)]
        )
        lb_mult[:, self.true_label] = 1
        ub_mult[:, self.true_label] = 1

        for i in range(torch.flatten(ub).size(0)):
            if i < self.true_label:
                lb_mult[i, i] = -1
                ub_mult[i, i] = -1
            elif i > self.true_label:
                lb_mult[i - 1, i] = -1
                ub_mult[i - 1, i] = -1

        bound = AlgebraicBound(
            ub_mult,
            lb_mult,
            torch.zeros(torch.flatten(ub).size(0) - 1),
            torch.zeros(torch.flatten(ub).size(0) - 1),
        )

        self.backward(bound)
        self.ub = bound.ub_bias
        self.lb = bound.lb_bias
        return Bound(lb=bound.lb_bias, ub=bound.ub_bias)

    def backward(self, bound: AlgebraicBound) -> None:
        """
        Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the current layer.
        Recomputes the parameters of the AlgebraicBound in-place and if the Verifier is not an InputVerifier, calls self.previous.backward!

        Args:
            bound: AlgebraicBound(ub_mult, lb_mult, ub_bias, lb_bias)
                ub_mult: Tensor: verifier_in_forward_pass.output_size x verifer.output_size
                ub_bias: Tensor: verifier_in_forward_pass.output_size

        Returns:
            None
        """
        self.previous.backward(bound)

    def reset(self):
        """Reset the Parameters of the verifier to their initial values."""
        print("No params to reset")
