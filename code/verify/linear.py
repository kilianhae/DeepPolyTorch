import torch
import numpy as np
from bound import Bound, AlgebraicBound
from verify.verify import Verifier, Bound, AlgebraicBound
from typing import Optional


class LinearVerifier(Verifier):
    """A Verifier for a torch.nn.Linear layer."""

    def __init__(
        self, layer: torch.nn.Linear, previous: Verifier, config: dict | None = None
    ):
        torch.nn.Module.__init__(self)
        self.previous = previous
        self.weights = layer.weight.detach()
        self.biases = layer.bias.detach()
        self.bound = None
        self._out_size = self.weights.size(0)
        self._out_dims = self.weights.size()

    def forward(self, x: Bound) -> Bound:
        """Calculates the numerical bounds of this layers output variables (by backwarding it through all previous layers) and returns them."""
        algebraic_bound = AlgebraicBound(
            self.weights, self.weights, self.biases, self.biases
        )
        self.previous.backward(algebraic_bound)
        self.bound = Bound(ub=algebraic_bound.ub_bias, lb=algebraic_bound.lb_bias)
        return self.bound

    def backward(self, bound: AlgebraicBound) -> None:
        """
        Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the current layer. 
        
        So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Then propagates the bounds to the previous layer.
        """
        bound.ub_bias = bound.ub_bias + bound.ub_mult @ self.biases
        bound.lb_bias = bound.lb_bias + bound.lb_mult @ self.biases
        bound.ub_mult = bound.ub_mult @ self.weights
        bound.lb_mult = bound.lb_mult @ self.weights
        self.previous.backward(bound)

    def reset(self):
        """Resets the Parameters of the verifier to their initial values."""
        print("No params to reset")


class Conv2DVerifier(Verifier):
    """A Verifier for a torch.nn.Conv2d layer."""

    def __init__(self, layer: torch.nn.Conv2d, previous: Verifier):
        torch.nn.Module.__init__(self)
        self.previous = previous
        self.kernel_size = layer.kernel_size
        self.in_channels = layer.in_channels
        self.out_channels = layer.out_channels
        self.padding = layer.padding
        self.stride = layer.stride
        self.dilation = layer.dilation

        self.mult = (
            None
        )  # type: Optional[torch.Tensor] # at this stage the multiplication tensor is not yet computed, as we dont know what input size the layer will have,
        self.bias = (
            None
        )  # type: Optional[torch.Tensor] # at this stage the bias tensor is not yet computed, as we dont know what input size the layer will have

        self.weights_unflattened = (
            layer.weight.detach()
        )  # type: torch.Tensor # shape: (out_channels, in_channels, kernel_size[0], kernel_size[1])
        self.weights = torch.flatten(
            layer.weight, start_dim=1
        ).detach()  # type: torch.Tensor # shape: (out_channels, in_channels * kernel_size[0] * kernel_size[1]) I THINK!
        if layer.bias is not None:
            self.biases = layer.bias.detach()
        else:
            self.biases = torch.zeros(self.weights.size(0)).detach()
        self.bound = None

        self._out_dims = self.compute_out_dims()
        self._out_size = int(torch.tensor(self._out_dims).prod().item())

    def compute_out_dims(self) -> torch.Size:
        """Computes the output dimensions of the conv layer."""
        weight_dimensions = self.weights_unflattened.size()
        assert isinstance(self.padding, tuple)
        out_channels = int(weight_dimensions[0])
        out_height = int(
            (
                self.previous.out_dims[1]
                + 2 * self.padding[0]
                - self.dilation[0] * (weight_dimensions[2] - 1)
                - 1
            )
            / self.stride[0]
            + 1
        )
        out_width = int(
            (
                self.previous.out_dims[2]
                + 2 * self.padding[1]
                - self.dilation[1] * (weight_dimensions[3] - 1)
                - 1
            )
            / self.stride[1]
            + 1
        )
        out_dims = torch.Size([out_channels, out_height, out_width])
        return out_dims

    def forward(self, x: Bound) -> Bound:
        """Calculates the numerical bounds of this layers output variables (by backwarding it through all previous layers) and returns them."""
        # check if the multiplication matrix is already implemented and if not compute it (this saves computation as it is only computed once and can be reused in future optimization steps or wehn running with different inputs)
        if self.mult is None or self.bias is None:
            self.mult, self.bias = self.compute_mult()
        algebraic_bound = AlgebraicBound(
            ub_mult=self.mult, lb_mult=self.mult, lb_bias=self.bias, ub_bias=self.bias
        )
        self.previous.backward(algebraic_bound)
        self.bound = Bound(ub=algebraic_bound.ub_bias, lb=algebraic_bound.lb_bias)
        return self.bound

    def compute_mult(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the matrix multiplication matrix and the bias vector for the conv layer.
        
        The mult tensor represents a matrix A and the bias tensor a vector b s.t. a vector l that respresents an affine expression (expression = l^T * y + bias) w.r.t to the ouputs of the conv layer (Conv(x)=y, assume x and y in flattened form),
        then the affine expression can be restated in terms of the inputs of the conv layer (x) as l' = A^T * l and bias' = bias + b^T * l, the the equivalent expresion is: expression = l'^T * y + bias'.
        They will be used in the backward method to compute the bounds of the previous layer.
        """

        padding = self.padding[0]
        stride = self.stride[0]
        kernel_size = self.kernel_size[0]
        in_channels = self.in_channels
        out_channels = self.out_channels

        assert isinstance(padding, int)
        # compute all dimensions
        in_channels = self.previous.out_dims[0]
        in_height = self.previous.out_dims[1]
        in_width = self.previous.out_dims[2]
        out_channels = self.out_dims[0]
        out_height = self.out_dims[1]
        out_width = self.out_dims[2]

        final_matrix = torch.zeros(
            (in_width * in_height * in_channels, out_width * out_height * out_channels)
        )
        flattened_weights = self.weights.flatten()

        # Check if kernel size is odd
        is_even_kernel = kernel_size % 2 == 0
        if is_even_kernel:
            offset = int(kernel_size / 2)
        else:
            offset = int((kernel_size - 1) / 2)

        kernel_size_kadenz = kernel_size * kernel_size
        output_image_kadenz = out_width * out_height
        final_matrix_col_indices = torch.zeros(out_width * out_height * out_channels)
        final_matrix_weight_indices = torch.zeros(flattened_weights.size())

        for channel_in_idx in range(in_channels):
            # Only iterate over the ones where the kernel fits
            top_bound_height = in_height - offset + padding
            top_bound_width = in_width - offset + padding
            if is_even_kernel:
                top_bound_height += 1
                top_bound_width += 1

            for col_in in range(offset - padding, top_bound_height, stride):
                for row_in in range(offset - padding, top_bound_width, stride):
                    # Indices of the output
                    col_out = int((col_in - offset + padding) / stride)
                    row_out = int((row_in - offset + padding) / stride)

                    # Final matrix column indices is the relative position inside the output image
                    final_matrix_col_indices.fill_(0)  # Resetting the tensor to zero
                    first_idx = (col_out) * out_width + row_out
                    num_elements_to_fill = (
                        len(final_matrix_col_indices) - first_idx
                    ) // (output_image_kadenz) + 1
                    fill_indices = first_idx + torch.arange(num_elements_to_fill) * (
                        output_image_kadenz
                    )
                    # Clamp fill_indices to the length of final_matrix_col_indices to avoid index out of bounds
                    fill_indices = fill_indices[
                        fill_indices < len(final_matrix_col_indices)
                    ]
                    # Fill in the tensor
                    final_matrix_col_indices[fill_indices] = 1
                    kernel_count_idx = 0
                    top_col_bound = col_in + offset + 1
                    top_row_bound = row_in + offset + 1
                    if is_even_kernel:
                        top_col_bound -= 1
                        top_row_bound -= 1
                    for kernel_col_idx in range(col_in - offset, top_col_bound):
                        for kernel_row_idx in range(row_in - offset, top_row_bound):
                            if (
                                kernel_col_idx < 0
                                or kernel_row_idx < 0
                                or kernel_col_idx >= in_height
                                or kernel_row_idx >= in_width
                            ):
                                kernel_count_idx += 1
                                continue

                            # Final matrix row index is the relative position inside the input image
                            final_matrix_row_idx = (
                                kernel_col_idx * in_width + kernel_row_idx
                            ) + (channel_in_idx * in_width * in_height)
                            # Final matrix weight indices is the relative position inside the kernel -> kernel_count_idx
                            # It has to be across the channels with kadenz of the kernel size
                            final_matrix_weight_indices.fill_(
                                0
                            )  # Resetting the tensor to zero
                            num_elements_to_fill = (
                                len(final_matrix_weight_indices) - kernel_count_idx
                            ) // (kernel_size_kadenz) + 1
                            fill_weight_indices = kernel_count_idx + torch.arange(
                                num_elements_to_fill
                            ) * (kernel_size_kadenz)
                            # Clamp fill_weight_indices to the length of final_matrix_weight_indices to avoid index out of bounds
                            fill_weight_indices = fill_weight_indices[
                                fill_weight_indices < len(final_matrix_weight_indices)
                            ]
                            # Fill in the tensor
                            final_matrix_weight_indices[fill_weight_indices] = 1

                            weight_mask = final_matrix_weight_indices == 1
                            # Select every in_channel'th element
                            weight_mask = torch.arange(len(weight_mask))[weight_mask]
                            weight_mask = weight_mask[channel_in_idx::in_channels]
                            # Fill the matrix
                            final_matrix[
                                final_matrix_row_idx, final_matrix_col_indices == 1
                            ] = flattened_weights[weight_mask]

                            kernel_count_idx += 1

        if self.biases is None:
            bias = torch.zeros(out_width * out_height * out_channels)
        else:
            bias = torch.repeat_interleave(self.biases, out_width * out_height)
        mult = final_matrix.transpose(0, 1)

        return mult, bias

    def backward(self, bound: AlgebraicBound) -> None:
        """
        Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the current layer. 
        
        So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer.
        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Then propagates the bounds to the previous layer.
        """
        bound.ub_bias = bound.ub_bias + bound.ub_mult @ self.bias
        bound.lb_bias = bound.lb_bias + bound.lb_mult @ self.bias
        bound.ub_mult = bound.ub_mult @ self.mult
        bound.lb_mult = bound.lb_mult @ self.mult

        return self.previous.backward(bound)

    def reset(self):
        """Resets the Parameters of the verifier to their initial values."""
        print("No params to reset")
