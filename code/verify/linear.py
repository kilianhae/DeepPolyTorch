import torch
import numpy as np
from bound import Bound, AlgebraicBound
from verify.verify import Verifier, Bound, AlgebraicBound
import time
class LinearVerifier(Verifier):
    """
    A Verifier for a torch.nn.Linear layer.
    """
    def __init__(self, layer: torch.nn.Linear, previous: Verifier):
        torch.nn.Module.__init__(self)
        self.previous = previous
        self.weights = layer.weight.detach()
        self.biases = layer.bias.detach()
        self.bound = None
        self._out_size = self.weights.size(0)
        self._out_dims = (self._out_size,)
        
    def forward(self, x: Bound) -> Bound:
        # create an identity matrix with dimensions of the output vector of the linear layer (equal to the first dim of the weight matrix)
        algebraic_bound = AlgebraicBound(torch.eye(self.out_size), torch.eye(self.out_size), torch.zeros(self.out_size), torch.zeros(self.out_size))
        self.backward(algebraic_bound)
        self.bound = Bound(ub=algebraic_bound.ub_bias, lb=algebraic_bound.lb_bias)
        return self.bound
    
    def backward(self, bound: AlgebraicBound) -> None:
        bound.ub_bias = bound.ub_bias + bound.ub_mult @ self.biases
        bound.lb_bias = bound.lb_bias + bound.lb_mult @ self.biases
        bound.ub_mult = bound.ub_mult @ self.weights
        bound.lb_mult = bound.lb_mult @ self.weights
        self.previous.backward(bound)


    

## NOTE: The following code is not used in the current implementation of DeepPoly. It is kept here for future reference. 
# class FlattenVerifier(Verifier):
#     """
#     Initiliazid in the forward method of a Transformer and passed backwards until the input variables at each step changing its algebraic representation.
#     """
#     def __init__(self, previous: Optional[Verifier]):
#         super().__init__(previous=previous)

#     def forward(self: Bound) -> Bound:
#         # here first we have to compute
#         lb, ub = self.previous.lb, self.previous.ub
#         bound = AlgebraicBound(torch.eye(torch.flatten(ub).size(0)), torch.eye(torch.flatten(ub).size(0)), torch.zeros(torch.flatten(ub).size(0)), torch.zeros(torch.flatten(ub).size(0)))
#         self.backward(bound)
#         self.ub = bound.ub_bias
#         self.lb = bound.lb_bias        
#         return self.next.forward()

#     def backward(self, bound: AlgebraicBound) -> None:
#         """
#         Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
#         Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
#         Then propagates the bounds to the previous layer.
#         """
#         bound.ub_mult = torch.reshape(bound.ub_mult, tuple(dim for dim in torch.concatenate((torch.tensor(bound.ub_mult.size(0))[None],torch.tensor(self.previous.ub.size())),dim=0)))
#         bound.lb_mult = torch.reshape(bound.lb_mult, tuple(dim for dim in torch.concatenate((torch.tensor(bound.lb_mult.size(0))[None],torch.tensor(self.previous.lb.size())),dim=0)))

#         # print(bound.ub_mult.size())
#         # MULT was of size eg: 100 x 100 but we need it to be 10 x 10 x 10 x 10
#         return self.previous.backward(bound)




class Conv2DVerifier(Verifier):
    """
    Initiliazid in the forward method of a Transformer and passed backwards until the input variables at each step changing its algebraic representation.
    """
    def __init__(self, layer: torch.nn.Conv2d, previous: Verifier):
        torch.nn.Module.__init__(self)
        self.previous = previous
        # for param in layer.parameters():
        #     param.requires_grad = False
        #self.layer = torch.jit.freeze(torch.jit.script(layer))
        self.kernel_size = layer.kernel_size
        self.in_channels = layer.in_channels
        self.out_channels = layer.out_channels
        self.padding = layer.padding
        self.stride = layer.stride
        self.dilation = layer.dilation


        self.weights_unflattened = layer.weight.detach() # type: torch.Tensor # shape: (out_channels, in_channels, kernel_size[0], kernel_size[1])
        self.weights = torch.flatten(layer.weight, start_dim=1).detach() # type: torch.Tensor # shape: (out_channels, in_channels * kernel_size[0] * kernel_size[1]) I THINK!
        if layer.bias is not None:
            self.biases = layer.bias.detach() 
        else:
            self.biases = torch.zeros(self.weights.size(0)).detach() 
        self.bound = None
       
       ## we stroe the dimesnions of the unflattened inputs and outputs of the conv layer for use on the backward or forward methods
        self._out_dims = self.compute_out_dims()
        self._out_size = int(torch.tensor(self._out_dims).prod().item())

        
        

    ## TODO: Check if this is correct!
    def compute_out_dims(self) -> tuple[int, ...]:
        """
        Computes the output dimensions of the conv layer given the input dimensions.
        """
        weight_dimensions = self.weights_unflattened.size()
        #assert isinstance(self.layer.padding, tuple)
        out_dims = tuple(torch.floor(torch.tensor([weight_dimensions[0],(self.previous.out_dims[1]+2*self.padding[0]-self.dilation[0]*(weight_dimensions[2]-1)-1)/self.stride[0] + 1, (self.previous.out_dims[2]+2*self.padding[1]-self.dilation[1]*(weight_dimensions[3]-1)-1)/self.stride[1] + 1])).int().tolist())
        return out_dims
        
    def forward(self, x: Bound) -> Bound:
        # create an identity matrix with dimensions of the output vector of the flattened conv layer (NOT equal to the first dim of the weight matrix)
        self.mult, self.bias = self.compute_mult()
        algebraic_bound = AlgebraicBound(torch.eye(self.out_size), torch.eye(self.out_size), torch.zeros(self.out_size), torch.zeros(self.out_size))
        self.backward(algebraic_bound)
        self.bound = Bound(ub=algebraic_bound.ub_bias, lb=algebraic_bound.lb_bias)
        return self.bound
    
    def compute_mult(self) -> tuple[torch.Tensor, torch.Tensor]:
        now = time.time()
        padding=self.padding[0]
        stride=self.stride[0]
        kernel_size=self.kernel_size[0]
        in_channels=self.in_channels
        out_channels=self.out_channels

        assert isinstance(padding, int)
        # compute all dimensions
        in_channels = self.previous.out_dims[0]
        in_height = self.previous.out_dims[1]
        in_width = self.previous.out_dims[2]
        out_channels = self.out_dims[0]
        out_height = self.out_dims[1]
        out_width = self.out_dims[2]

        

        final_matrix = torch.zeros((in_width * in_height * in_channels, out_width * out_height * out_channels))

        flattened_weights = self.weights.flatten()

        offset = int((kernel_size - 1) / 2)
        kernel_size_kadenz = kernel_size * kernel_size
        output_image_kadenz = out_width * out_height
        final_matrix_col_indices = torch.zeros(out_width * out_height * out_channels)
        final_matrix_weight_indices = torch.zeros(flattened_weights.size())

        for channel_in_idx in range(in_channels):
            # Only iterate over the ones where the kernel fits
            for col_in in range(offset-padding, in_height - offset + padding, stride):
                for row_in in range(offset-padding, in_width - offset + padding, stride):
                    # Indices of the output
                    col_out = int((col_in - offset + padding)/stride)
                    row_out = int((row_in - offset + padding)/stride)

                    #print("Channel: ", channel_in_idx, "Col: ", col_in, "Row: ", row_in, "Column out: ", col_out, "Row out: ", row_out)


                    # Final matrix column indices is the relative position inside the output image
                    final_matrix_col_indices.fill_(0)  # Resetting the tensor to zero
                    first_idx = (col_out) * out_width + row_out
                    num_elements_to_fill = (len(final_matrix_col_indices) - first_idx) // (output_image_kadenz) + 1
                    fill_indices = first_idx + torch.arange(num_elements_to_fill) * (output_image_kadenz)
                    # Clamp fill_indices to the length of final_matrix_col_indices to avoid index out of bounds
                    fill_indices = fill_indices[fill_indices < len(final_matrix_col_indices)]
                    # Fill in the tensor
                    final_matrix_col_indices[fill_indices] = 1

                    
                    kernel_count_idx = 0
                    for kernel_col_idx in range(col_in - offset, col_in + offset + 1):
                        for kernel_row_idx in range(row_in - offset, row_in + offset + 1):
                            if kernel_col_idx < 0 or kernel_row_idx < 0 or kernel_col_idx >= in_height or kernel_row_idx >= in_width:
                                kernel_count_idx += 1
                                continue
                    
                            # Final matrix row index is the relative position inside the input image
                            final_matrix_row_idx = (kernel_col_idx * in_width + kernel_row_idx) + (channel_in_idx * in_width * in_height)
                            #print("Final matrix row idx: ", final_matrix_row_idx)

                            # Final matrix weight indices is the relative position inside the kernel -> kernel_count_idx
                            # It has to be across the channels with kadenz of the kernel size
                            final_matrix_weight_indices.fill_(0)  # Resetting the tensor to zero
                            num_elements_to_fill = (len(final_matrix_weight_indices) - kernel_count_idx) // (kernel_size_kadenz) + 1
                            fill_weight_indices = kernel_count_idx + torch.arange(num_elements_to_fill) * (kernel_size_kadenz)
                            # Clamp fill_weight_indices to the length of final_matrix_weight_indices to avoid index out of bounds
                            fill_weight_indices = fill_weight_indices[fill_weight_indices < len(final_matrix_weight_indices)]
                            # Fill in the tensor
                            final_matrix_weight_indices[fill_weight_indices] = 1

                            weight_mask = final_matrix_weight_indices == 1
                            # Select every in_channel'th element
                            weight_mask = torch.arange(len(weight_mask))[weight_mask]
                            weight_mask = weight_mask[channel_in_idx::in_channels]
                            # Fill the matrix
                            final_matrix[final_matrix_row_idx, final_matrix_col_indices == 1] = flattened_weights[weight_mask]

                            kernel_count_idx += 1

        
        #in_width_p = in_width + padding * 2
        #in_height_p = in_height + padding * 2
        #weights = self.weights_unflattened

        #size_p = in_height_p * in_width_p
        #in_dim = size_p * in_channels
        #out_dim = out_height * out_width * out_channels
        #res = torch.zeros((out_dim, in_dim))

        # build row fillers
        #len_rows = (in_channels - 1) * size_p + (kernel_size - 1) * in_width_p + kernel_size
        #channels = torch.zeros((out_channels, len_rows))

        # for i_out in range(out_channels):
        #     for i_in in range(in_channels):
        #         i_p = i_in * size_p
        #         for k in range(kernel_size):
        #             start = i_p + k * in_width_p
        #             end = start + kernel_size
        #             channels[i_out, start:end] = weights[i_out, i_in, k]

        #     for i_out_height in range(out_height):
        #         for i_out_width in range(out_width):
        #             start = i_out_height * stride * in_width_p + i_out_width * stride
        #             end = start + len_rows
        #             output = i_out * out_height * out_width + i_out_height * out_width + i_out_width
        #             res[output, start:end] = channels[i_out]

        # # remove padding
        # padding_rows = []
        # for i_in in range(in_channels):
        #     for i_in_height in range(in_height_p):
        #         for i_in_width in range(in_width_p):
        #             if i_in_width < padding or i_in_width >= padding + in_width:
        #                 padding_rows.append(i_in * size_p + i_in_height * in_width_p + i_in_width)

        #         if i_in_height < padding or i_in_height >= padding + in_height:
        #             start = i_in * size_p + i_in_height * in_width_p
        #             end = start + in_width_p
        #             padding_rows = padding_rows + list(range(start, end))

        # padding_rows = list(np.unique(np.array(padding_rows)))  # delete duplicates

        # lc = torch.from_numpy(np.delete(res.numpy(), padding_rows, axis=1)).detach()

        if self.biases is None:
            ret_bias = torch.zeros(out_width * out_height * out_channels)
        else:
            ret_bias = torch.repeat_interleave(self.biases, out_width * out_height)
        print("time to compute mult: ", time.time()-now)
        return final_matrix.transpose(0,1), ret_bias

    def backward(self, bound: AlgebraicBound) -> None:
        """
        Input is a AlgebraicBound object that represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the 
        current layer. So the contents are tensors of the shape: Tensor: number of out-neurons in initializing layer x number of out-neurons in current layer
        Recomputes the bounds so that it represents the algebraic bounds of the initializing layer w.r.t. to the output neurons of the previous layer.
        Then propagates the bounds to the previous layer.
        """
        now = time.time()
        # parameters
        bound.ub_bias = bound.ub_bias + bound.ub_mult @ self.bias
        bound.lb_bias = bound.lb_bias + bound.lb_mult @ self.bias
        bound.ub_mult = bound.ub_mult @ self.mult
        bound.lb_mult = bound.lb_mult @ self.mult
        # padding=self.padding[0]
        # stride=self.stride[0]
        # kernel_size=self.kernel_size[0]
        # in_channels=self.in_channels
        # out_channels=self.out_channels
        # weights = self.weights_unflattened
        # out_channels = self.out_dims[0]
        # out_height = self.out_dims[1]
        # out_width = self.out_dims[2]
        # in_height = self.previous.out_dims[1]
        # in_width = self.previous.out_dims[2]
        
        # weights = self.weights_unflattened
        # assert isinstance(padding, int) 
        # padded_size = list(self.previous.out_dims)
        # padded_size = [bound.lb_bias.size(0)] + padded_size
        # padded_size[2] = padded_size[2] + 2 * padding
        # padded_size[3] = padded_size[3] + 2 * padding
        # bound.ub_mult = torch.reshape(bound.ub_mult, [bound.lb_bias.size(0)]+list(self.out_dims))
        # bound.lb_mult = torch.reshape(bound.lb_mult, [bound.lb_bias.size(0)]+list(self.out_dims))
        # im_ub = torch.zeros(padded_size)
        # im_lb = torch.zeros(padded_size)
        # for i_out in range(out_channels):
        #     channel_weights = weights[i_out]
        #     for i_h in range(out_height):
        #         for i_w in range(out_width):
        #             el_ub = bound.ub_mult[:,i_out,i_h,i_w]
        #             el_lb = bound.lb_mult[:,i_out,i_h,i_w]
        #             now1 = time.time()
        #             im_ub[:,:,stride*i_h:stride*i_h+kernel_size, stride*i_w:stride*i_w+kernel_size] = im_ub[:,:,stride*i_h:stride*i_h+kernel_size, stride*i_w:stride*i_w+kernel_size] + el_ub.view(-1,1,1,1) * channel_weights
        #             im_lb[:,:,stride*i_h:stride*i_h+kernel_size, stride*i_w:stride*i_w+kernel_size] = im_lb[:,:,stride*i_h:stride*i_h+kernel_size, stride*i_w:stride*i_w+kernel_size] + el_lb.view(-1,1,1,1) * channel_weights
        #             print(time.time() - now1)
        # if padding > 0:
        #     im_ub = im_ub[:,:,padding:-padding,padding:-padding]
        #     im_lb = im_lb[:,:,padding:-padding,padding:-padding]

        # sum_ub = bound.ub_mult.sum(dim=[2,3])
        # sum_lb = bound.lb_mult.sum(dim=[2,3])
        # bias_r = self.biases.view(1,-1)
        # bound.ub_bias = bound.ub_bias + (sum_ub * bias_r).sum(1)
        # bound.lb_bias = bound.lb_bias + (sum_lb * bias_r).sum(1)

        # bound.ub_mult = im_ub.reshape(im_ub.size(0),-1)
        # bound.lb_mult = im_lb.reshape(im_ub.size(0),-1)
        print("time to compute backward: ", time.time()-now)

        return self.previous.backward(bound)