"""An implementation of a Class that loads a Verifier for a given networks, optimizes the parameters and outputs wether the network is robust within a given Domain."""
from __future__ import annotations
import torch
from bound import Bound
from verify.verify import Verifier, FinalLossVerifier, InputVerifier
from verify.linear import LinearVerifier, Conv2DVerifier
from verify.activation import (
    ReluVerifier,
    LeakyReluVerifierSteep,
)

OPT_DICT = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "Adagrad": torch.optim.Adagrad,
    "Adadelta": torch.optim.Adadelta,
    "RMSprop": torch.optim.RMSprop,
}


class DeepPoly(torch.nn.Module):
    """
    An object that represents a DeepPoly Verifier for a given network.

    It is initialized with a model and a true label and then can be called with an input
    and an epsilon to compute the bounds and to optimize on ReLu parameters to get tight and sound bounds.

    Attributes
    ----------
    model : torch.nn.Sequential representing the model to be verified
    true_label : int representing the true label of the input
    verifiers : torch.nn.Sequential representing the verifiers for each layer of the model
    input_verifier : InputVerifier representing the verifier for the input layer
    """

    def __init__(
        self,
        model: torch.nn.Sequential,
        input: torch.Size,
        true_label: int,
        config: dict,
    ):
        super().__init__()

        self.config = config

        verifiers = []  # type: list[Verifier]
        self.input_verifier = InputVerifier(input)
        verifiers.append(self.input_verifier)

        for module in model:
            if isinstance(module, torch.nn.Linear):
                verifiers.append(LinearVerifier(layer=module, previous=verifiers[-1]))
            elif isinstance(module, torch.nn.Conv2d):
                verifiers.append(Conv2DVerifier(layer=module, previous=verifiers[-1]))
            elif isinstance(module, torch.nn.ReLU):
                verifiers.append(ReluVerifier(previous=verifiers[-1], **config))
            elif isinstance(module, torch.nn.LeakyReLU):
                verifiers.append(
                    LeakyReluVerifierSteep(
                        negative_slope=module.negative_slope,
                        previous=verifiers[-1],
                        **config,
                    )
                )
            elif isinstance(module, torch.nn.Flatten):
                pass
            else:
                raise NotImplementedError

        output_verifier = FinalLossVerifier(verifiers[-1], true_label)
        verifiers.append(output_verifier)
        self.verifiers = torch.nn.Sequential(*verifiers)

    def reset(self) -> None:
        """Reset the fitted Parameters to their init values. This should be used whenever you train want to use a new Input sample, as the parameters are input-specific."""
        for verifier in self.verifiers:
            verifier.reset()  # type: ignore # need this since loop through sequential is typed as torch.Module and not as Verifier

    def forward(self, input_bound: Bound) -> Bound:
        """Given a Bound object representing the precodition Domain to check for robustness, return an output Bound representing at each index the difference between the max logprob and the logprob of the true class."""
        final_bound = self.verifiers.forward(self.input_verifier.forward(input_bound))
        return final_bound

    def verify(self, x: torch.Tensor, eps: float, epochs: int = 32) -> bool:
        """Given an input image and a eps (indicating the max deviation on a single dimension a.k.a and upper bound to L_infinity), computes wether the network is robust within this Domain.

        To do so it will try to fit the DeePoly Parameters over a given nr of epochs by minimizing using Projected Gradient Descent with Adam.

        Args:
            x: Tensor representing the input image
            eps: float representing the max deviation on a single dimension a.k.a and upper bound to L_infinity
            epochs: int representing the number of epochs to train the verifier

        Returns:
            bool: True if the network is robust within the given Domain, False otherwise
        """
        ub_in = torch.Tensor.clamp(x + eps, min=0, max=1)
        lb_in = torch.Tensor.clamp(x - eps, min=0, max=1)
        input_bound = Bound(ub=ub_in, lb=lb_in)

        if len(list(self.verifiers.parameters())) != 0:
            opt = OPT_DICT[self.config["optimizer"]["name"]](
                list(self.verifiers.parameters()), **self.config["optimizer"]["params"]
            )
            for i in range(0, epochs):
                opt.zero_grad()
                final_bound = self.forward(input_bound)
                lowest = torch.min(final_bound.lb)
                loss = torch.sum(-final_bound.lb)

                if lowest >= 0:
                    print("stopped at iteration: ", i)
                    break

                loss.backward()
                opt.step()
                if self.config["activation"]["standartization"] == "clamp":
                    with torch.no_grad():
                        for alpha in list(self.verifiers.parameters()):
                            alpha.clamp_(min=0, max=1)
        else:
            final_bound = self.forward(input_bound)
        if torch.min(final_bound.lb) >= 0:  # type: ignore
            return True
        else:
            return False
