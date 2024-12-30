import torch

# quantization
def get_q_scale_symmetric(tensor, dtype=torch.int8):
    r_max = tensor.max().item() 
    q_max = torch.iinfo(dtype).max
    return r_max / q_max
test_tensor = torch.randn((4,4))
print(get_q_scale_symmetric(test_tensor))

from helper import linear_q_with_scale_and_zero_point

def linear_q_symmetric(tensor, dtype=torch.int8):
    scale = get_q_scale_symmetric(tensor)
    # in symmetric quantization, zero point is 0
    quantized_tensor = linear_q_with_scale_and_zero_point(tensor, scale, 0, dtype)
    return quantized_tensor, scale
quantized_tensor, scale = linear_q_symmetric(test_tensor)
print(quantized_tensor)

# dequantization

from helper import linear_dequantization, plot_quantization_errors
from helper import quantization_error

dequatizated_tensor = linear_dequantization(quantized_tensor, scale, 0)
plot_quantization_errors(test_tensor, quantized_tensor, dequatizated_tensor)

print(f"""Quantization Error : \
{quantization_error(test_tensor, dequatizated_tensor)}""")