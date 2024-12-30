# compress any model in 8-bit precision
# w8_a16_forword


import torch
import torch.nn as nn
import torch.nn.functional as F

random_int8 = torch.randint(-128, 127, (32, 16)).to(torch.int8)
random_hs = torch.randn((1,16), dtype=torch.bfloat16)
scales = torch.randn((1,32), dtype=torch.bfloat16)
bias = torch.randn((1,32), dtype=torch.bfloat16)

print(F.linear(random_hs, random_int8.to(random_hs.dtype))*scales)
random_int8 = (random_int8.to(random_hs.dtype).t() * scales).t()
print(F.linear(random_hs, random_int8))

# input forward with quanted weight
def w8_a16_forward(weight, input, scales, bias=None):
    casted_weights = weight.to(input.dtype)
    output = F.linear(input, casted_weights) * scales
    if bias is not None:
        output = output + bias
    return output

class W8A16LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, 
                 bias=True, dtype=torch.float32):
        super().__init__()
        self.register_buffer(
            "int8_weights",
            torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8
            )
        )
        self.register_buffer("scales", torch.randn((out_features), dtype=dtype))
        if bias:
            self.register_buffer("bias", torch.randn((1, out_features), dtype=dtype))
        else:
            self.bias = None
    def forward(self, input):
        return w8_a16_forward(self.int8_weights, input, self.scales, self.bias)
    def quantize(self, weights):
        w_fp32 = weights.clone().to(torch.float32)
        scales = w_fp32.abs().max(dim=-1).values / 127
        scales = scales.to(weights.dtype)
        int8_weights = torch.round(weights/scales.unsqueeze(1)).to(torch.int8)
        self.int8_weights = int8_weights
        self.scales = scales

module = W8A16LinearLayer(4, 8)
random_matrix = torch.randn((4, 8), dtype=torch.bfloat16)
module.quantize(random_matrix)