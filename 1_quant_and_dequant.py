import torch
# quantization with random scale and zero point
def linear_q_with_scale_and_zero_point(
        tensor, scale, zero_point, dtype = torch.int8):
    # q = int(round(r/s+z))
    scaled_and_shifted_tensor = tensor / scale + zero_point
    rounded_tensor = torch.round(scaled_and_shifted_tensor)
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max
    q_tensor = rounded_tensor.clamp(q_min, q_max).to(dtype)
    return q_tensor
test_tensor=torch.tensor(
    [[191.6, -13.5, 728.6],
     [92.14, 295.5,  -184],
     [0,     684.6, 245.5]]
)
scale = 3.5
zero_point = -70
print(test_tensor)
q_tensor = linear_q_with_scale_and_zero_point(test_tensor, scale, zero_point)
print(q_tensor)

# dequantization with random scale and zero point
# tensor=scale*(quanted_tensor-zero_point)
dequantized_tensor = scale * (q_tensor.float() - zero_point)
print(dequantized_tensor)