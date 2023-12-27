import torch
from verify.verify import Verifier, Bound, AlgebraicBound


class ReluVerifier(Verifier):
    """
    A Verifier for a torch.nn.ReLU layer.
    """

    def __init__(self, previous: Verifier, **config):
        torch.nn.Module.__init__(self)
        self.previous = previous
        self._out_size = self.previous.out_size
        self._out_dims = self.previous.out_dims

        self.standartization = config.get("standartization", "clamp")
        self.alpha_init = config.get("init", 0.5)
        self._init_activations()

    def _init_activations(self) -> None:
        """Initialize the alpha parameters of the layer."""
        if isinstance(self.alpha_init, int | float):
            self.alpha = torch.nn.Parameter(
                torch.zeros(self.out_size) + self.alpha_init
            ).requires_grad_(True)
        elif self.alpha_init == "random":
            self.alpha = torch.nn.Parameter(torch.rand(self.out_size)).requires_grad_(
                True
            )
        elif self.alpha_init == "normal":
            self.alpha = torch.nn.Parameter(
                torch.clamp(
                    torch.normal(torch.zeros(self.out_size) + 0.5, 0.25), min=0, max=1
                )
            ).requires_grad_(True)
        else:
            raise NotImplementedError

    def forward(self, x: Bound) -> Bound:
        """Calculates the numerical bounds of this layers output variables (by backwarding it through all previous layers) and returns them."""
        lb, ub = x.lb, x.ub
        self.slope = torch.clamp(ub / (ub - lb), min=0)
        self.ub_mult = torch.diag(torch.where(lb > 0, 1.0, self.slope))

        if self.standartization == "sigmoid":
            normalized_alphas = torch.where(lb > 0, 1.0, torch.sigmoid(self.alpha))
        elif self.standartization == "clamp":
            normalized_alphas = torch.where(lb > 0, 1.0, self.alpha)
        else:
            raise NotImplementedError

        self.lb_mult = torch.diag(torch.where(ub < 0, 0, normalized_alphas))
        self.ub_bias = torch.where(lb > 0, 0, (-self.slope * lb))
        self.lb_bias = torch.zeros_like(lb)

        bound = AlgebraicBound(self.ub_mult, self.lb_mult, self.ub_bias, self.lb_bias)
        self.previous.backward(bound)
        self.ub = bound.ub_bias
        self.lb = bound.lb_bias
        return Bound(lb=self.lb, ub=self.ub)

    def backward(self, bound: AlgebraicBound) -> None:
        """
        Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer.

        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Then propagates the bounds to the previous layer.
        """
        bound.ub_bias = (
            bound.ub_bias
            + torch.where(bound.ub_mult > 0, bound.ub_mult, 0) @ self.ub_bias
            + torch.where(bound.ub_mult < 0, bound.ub_mult, 0) @ self.lb_bias
        )
        bound.lb_bias = (
            bound.lb_bias
            + torch.where(bound.lb_mult > 0, bound.lb_mult, 0) @ self.lb_bias
            + torch.where(bound.lb_mult < 0, bound.lb_mult, 0) @ self.ub_bias
        )
        bound.ub_mult = (
            torch.where(bound.ub_mult > 0, bound.ub_mult, 0) @ self.ub_mult
            + torch.where(bound.ub_mult < 0, bound.ub_mult, 0) @ self.lb_mult
        )
        bound.lb_mult = (
            torch.where(bound.lb_mult > 0, bound.lb_mult, 0) @ self.lb_mult
            + torch.where(bound.lb_mult < 0, bound.lb_mult, 0) @ self.ub_mult
        )
        self.previous.backward(bound)

    def reset(self):
        """Resets the Parameters of the verifier to their initial values."""
        self._init_activations()


class LeakyReluVerifierFlat(Verifier):
    """
    A Verifier for a torch.nn.LeakyReLU layer with negative slope < 1.
    """

    def __init__(self, negative_slope: float, previous: Verifier, **config):
        torch.nn.Module.__init__(self)
        self.previous = previous
        self.negative_slope = negative_slope
        self._out_size = self.previous.out_size
        self._out_dims = self.previous.out_dims

        self.standartization = config.get("standartization", "clamp")
        self.alpha_init = config.get("init", 0.5)
        self._init_activations()

    def _init_activations(self) -> None:
        """Initialize the alpha parameters of the layer."""
        if isinstance(self.alpha_init, int | float):
            self.alpha = torch.nn.Parameter(
                torch.zeros(self.out_size) + self.alpha_init
            ).requires_grad_(True)
        elif self.alpha_init == "random":
            self.alpha = torch.nn.Parameter(torch.rand(self.out_size)).requires_grad_(
                True
            )
        elif self.alpha_init == "normal":
            self.alpha = torch.nn.Parameter(
                torch.clamp(
                    torch.normal(torch.zeros(self.out_size) + 0.5, 0.25), min=0, max=1
                )
            ).requires_grad_(True)
        else:
            raise NotImplementedError

    def forward(self, x: Bound) -> Bound:
        """Calculates the numerical bounds of this layers output variables (by backwarding it through all previous layers) and returns them."""
        # here first we have to compute
        lb, ub = x.lb, x.ub
        # need to clamp the slope so we dont compute negative slopes

        self.slope = (ub - self.negative_slope * lb) / (ub - lb)
        self.ub_mult = torch.where(lb > 0, 1.0, self.slope)
        self.ub_mult = torch.diag(
            torch.where(ub < 0, self.negative_slope, self.ub_mult)
        )

        if self.standartization == "sigmoid":
            normalized_alphas = (
                torch.where(lb > 0, 1.0, torch.sigmoid(self.alpha))
                * (1 - self.negative_slope)
                + self.negative_slope
            )
        elif self.standartization == "clamp":
            normalized_alphas = (
                torch.where(lb > 0, 1.0, self.alpha) * (1 - self.negative_slope)
                + self.negative_slope
            )

        else:
            raise NotImplementedError
        self.lb_mult = torch.where(lb > 0, 1.0, normalized_alphas)
        self.lb_mult = torch.diag(
            torch.where(ub < 0, self.negative_slope, self.lb_mult)
        )

        self.ub_bias = torch.where(lb > 0, 0, (self.negative_slope - self.slope) * lb)
        self.ub_bias = torch.where(ub < 0, 0, self.ub_bias)
        self.lb_bias = torch.zeros_like(lb)

        bound = AlgebraicBound(self.ub_mult, self.lb_mult, self.ub_bias, self.lb_bias)
        self.previous.backward(bound)
        self.ub = bound.ub_bias
        self.lb = bound.lb_bias
        return Bound(lb=self.lb, ub=self.ub)

    def backward(self, bound: AlgebraicBound) -> None:
        """
        Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the
        current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Then propagates the bounds to the previous layer.
        """
        bound.ub_bias = (
            bound.ub_bias
            + torch.where(bound.ub_mult > 0, bound.ub_mult, 0) @ self.ub_bias
            + torch.where(bound.ub_mult < 0, bound.ub_mult, 0) @ self.lb_bias
        )
        bound.lb_bias = (
            bound.lb_bias
            + torch.where(bound.lb_mult > 0, bound.lb_mult, 0) @ self.lb_bias
            + torch.where(bound.lb_mult < 0, bound.lb_mult, 0) @ self.ub_bias
        )

        bound.ub_mult = (
            torch.where(bound.ub_mult > 0, bound.ub_mult, 0) @ self.ub_mult
            + torch.where(bound.ub_mult < 0, bound.ub_mult, 0) @ self.lb_mult
        )
        bound.lb_mult = (
            torch.where(bound.lb_mult > 0, bound.lb_mult, 0) @ self.lb_mult
            + torch.where(bound.lb_mult < 0, bound.lb_mult, 0) @ self.ub_mult
        )

        self.previous.backward(bound)

    def reset(self):
        """Resets the Parameters of the verifier to their initial values."""
        self._init_activations()


class LeakyReluVerifierSteep(Verifier):
    """
    A Verifier for a torch.nn.LeakyReLU layer with negative slope > 1.
    """

    def __init__(self, negative_slope: float, previous: Verifier, **config):
        torch.nn.Module.__init__(self)
        self.previous = previous
        self.negative_slope = negative_slope
        self._out_size = self.previous.out_size
        self._out_dims = self.previous.out_dims

        self.standartization = config.get("standartization", "clamp")
        self.alpha_init = config.get("init", 0.5)
        self._init_activations()

    def _init_activations(self) -> None:
        """Initialize the alpha parameters of the layer."""
        if isinstance(self.alpha_init, int | float):
            self.alpha = torch.nn.Parameter(
                torch.zeros(self.out_size) + self.alpha_init
            ).requires_grad_(True)
        elif self.alpha_init == "random":
            self.alpha = torch.nn.Parameter(torch.rand(self.out_size)).requires_grad_(
                True
            )
        elif self.alpha_init == "normal":
            self.alpha = torch.nn.Parameter(
                torch.clamp(
                    torch.normal(torch.zeros(self.out_size) + 0.5, 0.25), min=0, max=1
                )
            ).requires_grad_(True)
        else:
            raise NotImplementedError

    def forward(self, x: Bound) -> Bound:
        """Calculates the numerical bounds of this layers output variables (by backwarding it through all previous layers) and returns them."""
        # here first we have to compute
        lb, ub = x.lb, x.ub
        slope = (ub - self.negative_slope * lb) / (ub - lb)

        # need to clamp the slope so we dont compute negative slopes
        if self.standartization == "sigmoid":
            normalized_alphas = (
                torch.where(lb > 0, 1.0, torch.sigmoid(self.alpha))
                * (self.negative_slope - 1)
                + 1.0
            )
        elif self.standartization == "clamp":
            normalized_alphas = (
                torch.where(lb > 0, 1.0, self.alpha) * (self.negative_slope - 1) + 1.0
            )
        else:
            raise NotImplementedError

        # compute the elementwise upper and lower bound expressions and store them for future backward passes
        self.ub_mult = torch.where(lb > 0, 1.0, normalized_alphas)
        self.ub_mult = torch.diag(
            torch.where(ub < 0, self.negative_slope, self.ub_mult)
        )
        self.lb_mult = torch.where(lb > 0, 1.0, slope)
        self.lb_mult = torch.diag(
            torch.where(ub < 0, self.negative_slope, self.lb_mult)
        )

        self.lb_bias = torch.where(lb > 0, 0, (self.negative_slope - slope) * lb)
        self.lb_bias = torch.where(ub < 0, 0, self.lb_bias)
        self.ub_bias = torch.zeros_like(ub)

        bound = AlgebraicBound(self.ub_mult, self.lb_mult, self.ub_bias, self.lb_bias)
        self.previous.backward(bound)

        self.ub = bound.ub_bias
        self.lb = bound.lb_bias
        return Bound(lb=self.lb, ub=self.ub)

    def backward(self, bound: AlgebraicBound) -> None:
        """
        Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the
        current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Then propagates the bounds to the previous layer.
        """
        bound.ub_bias = (
            bound.ub_bias
            + torch.where(bound.ub_mult > 0, bound.ub_mult, 0) @ self.ub_bias
            + torch.where(bound.ub_mult < 0, bound.ub_mult, 0) @ self.lb_bias
        )
        bound.lb_bias = (
            bound.lb_bias
            + torch.where(bound.lb_mult > 0, bound.lb_mult, 0) @ self.lb_bias
            + torch.where(bound.lb_mult < 0, bound.lb_mult, 0) @ self.ub_bias
        )
        bound.ub_mult = (
            torch.where(bound.ub_mult > 0, bound.ub_mult, 0) @ self.ub_mult
            + torch.where(bound.ub_mult < 0, bound.ub_mult, 0) @ self.lb_mult
        )
        bound.lb_mult = (
            torch.where(bound.lb_mult > 0, bound.lb_mult, 0) @ self.lb_mult
            + torch.where(bound.lb_mult < 0, bound.lb_mult, 0) @ self.ub_mult
        )
        self.previous.backward(bound)

    def reset(self):
        """Resets the Parameters of the verifier to their initial values."""
        self._init_activations()
