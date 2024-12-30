import torch
from helper import linear_q_with_scale_and_zero_point, linear_dequantization, plot_quantization_errors

### a dummy tensor to test the implementation
test_tensor=torch.tensor(
    [[191.6, -13.5, 728.6],
     [92.14, 295.5,  -184],
     [0,     684.6, 245.5]]
)

q_min = torch.iinfo(torch.int8).min
q_max = torch.iinfo(torch.int8).max

r_min = test_tensor.min().item()
r_max = test_tensor.max().item()

# get scale and zero point  
scale = (r_max-r_min) / (q_max-q_min)
zero_point = int(round(q_min - r_min / scale))

# per-tensor quantization
def get_q_scale_and_zero_point(tensor, dtype=torch.int8):
    q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    r_min, r_max = tensor.min().item(), tensor.max().item()
    scale = (r_max-r_min) / (q_max-q_min)
    zero_point = q_min - r_min / scale
    if zero_point < q_min:
        zero_point = q_min
    elif zero_point > q_max:
        zero_point = q_max
    else:
        zero_point = int(round(zero_point))
    return scale, zero_point

new_scale, new_zero_point = get_q_scale_and_zero_point(test_tensor)
print(f"scale: {scale}, zero_point: {zero_point}")

quantized_tensor = linear_q_with_scale_and_zero_point(
    test_tensor, new_scale, new_zero_point, dtype=torch.int8)

print(f"quantized tensor:{quantized_tensor}, with scale:{new_scale}, zero_point:{new_zero_point}")
dequanted_tensor = linear_dequantization(quantized_tensor, new_scale, new_zero_point)
print(f"origin_tensor:{test_tensor}, dequantized tensor:{dequanted_tensor}")

plot_quantization_errors(test_tensor, quantized_tensor, dequanted_tensor)

def linear_quantization(tensor, dtype=torch.int8):
    scale, zero_point = get_q_scale_and_zero_point(tensor, dtype)
    quantized_tensor = linear_q_with_scale_and_zero_point(
        tensor, scale, zero_point, dtype)
    return quantized_tensor, scale, zero_point

r_tensor = torch.randn(4, 4)
quantized_tensor, scale, zero_point = linear_quantization(r_tensor)
dequanted_tensor = linear_dequantization(quantized_tensor, scale, zero_point)
plot_quantization_errors(r_tensor, quantized_tensor, dequanted_tensor)
print(f"squre mean:{(dequanted_tensor-r_tensor).square().mean()}")