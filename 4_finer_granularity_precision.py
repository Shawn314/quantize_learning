import torch

from helper import linear_q_symmetric, get_q_scale_symmetric, linear_dequantization, linear_q_with_scale_and_zero_point
from helper import plot_quantization_errors, quantization_error, linear_q_symmetric_per_channel

test_tensor=torch.randn((6,6))

# per_tensor, for simplicity, perform symmetric quantization

quanted_tensor, scale = linear_q_symmetric(test_tensor)
dequanted_tensor = linear_dequantization(quanted_tensor, scale, 0)

# plot_quantization_errors(test_tensor, quanted_tensor,
#                          dequanted_tensor)
print(f"""Quantization Error : \
{quantization_error(test_tensor, dequanted_tensor)}""")

# per_channel quantization

dim = 0 # 0 for row, 1 for column
output_dim = test_tensor.shape[dim]
scale = torch.zeros(output_dim)

# iterate through each row to calculate its scale
for index in range(output_dim):
    sub_tensor = test_tensor.select(dim, index)
    scale[index] = get_q_scale_symmetric(sub_tensor)

# get test tensor dim size
scale_shape = [1] * test_tensor.dim()
scale_shape[dim] = -1
scale = scale.view(scale_shape)
copy_scale = scale

quantized_tensor = linear_q_with_scale_and_zero_point(test_tensor, scale, 0)
print(quantized_tensor)

quanted_tensor_0, scale_0 = linear_q_symmetric_per_channel(test_tensor, dim=0)
quanted_tensor_1, scale_1 = linear_q_symmetric_per_channel(test_tensor, dim=1)

dequantized_tensor_0 = linear_dequantization(
    quanted_tensor_0, scale_0, 0)

plot_quantization_errors(
    test_tensor, quanted_tensor_0, dequantized_tensor_0)

print(f"""Quantization Error : \
{quantization_error(test_tensor, dequantized_tensor_0)}""")

dequantized_tensor_1 = linear_dequantization(
    quanted_tensor_1, scale_1, 0)

plot_quantization_errors(
    test_tensor, quanted_tensor_1, dequantized_tensor_1, n_bits=8)

print(f"""Quantization Error : \
{quantization_error(test_tensor, dequantized_tensor_1)}""")

# per-group
def linear_q_symmetric_per_group(tensor, group_size,
                                 dtype=torch.int8):
    t_shape = tensor.shape
    assert t_shape[1] % group_size == 0
    assert tensor.dim() == 2
    tensor = tensor.view(-1, group_size)
    quanted_tensor, scale = linear_q_symmetric_per_channel(tensor, dim=0, dtype=dtype)
    quanted_tensor = quanted_tensor.view(t_shape)
    return quanted_tensor, scale

def linear_dequantization_per_group(quanted_tensor, scale, group_size):
    t_shape = quanted_tensor.shape
    quanted_tensor = quanted_tensor.view(-1, group_size)
    dequantized_tensor = linear_dequantization(quanted_tensor, scale, 0)
    dequantized_tensor = dequantized_tensor.view(t_shape)
    return dequantized_tensor

group_size = 6
quanted_tensor, scale = linear_q_symmetric_per_group(test_tensor, group_size)
dequantized_tensor = linear_dequantization_per_group(quanted_tensor, scale, group_size)
plot_quantization_errors(test_tensor, quanted_tensor, dequantized_tensor)
print(f"""Quantization Error : \
{quantization_error(test_tensor, dequantized_tensor)}""")


# inferance

def quantized_linear_W8A32_without_bias(input, q_w, s_w, z_w):
    assert input.dtype == torch.float32
    assert q_w.dtype == torch.int8

    dequanted_weight = q_w.to(torch.float32) * s_w + z_w
    output = torch.nn.functional.linear(input, dequanted_weight)
    return output

input = torch.tensor([1,2,3], dtype=torch.float32)
weight = torch.randn((3,3))

q_w, s_w = linear_q_symmetric_per_group(weight, 3)
output = quantized_linear_W8A32_without_bias(input, q_w, s_w, 0)
print(f"This is the W8A32 output: {output}")

fp32_output = torch.nn.functional.linear(input, weight)
print(f"This is the output if we don't quantize: {fp32_output}")