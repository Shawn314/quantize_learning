import torch
import torch.nn as nn

from helper import W8A16LinearLayer

# Replace all  torch.nn.linear with W8A16LinearLayer layer
# Call quantize on the linear layers using the original weights

# model in-place linear layer replacement
def replace_linear_with_target(module, target_class, module_name_to_exclude):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not \
        any([x == name for x in module_name_to_exclude]):
            old_bias = child.bias
            new_module = target_class(child.in_features, 
                                      child.out_features, 
                                      old_bias is not None, 
                                      child.weight.dtype)
            setattr(module, name, new_module)
            if old_bias is not None:
                getattr(module, name).bias = old_bias
        else:
            replace_linear_with_target(child, 
                     target_class, module_name_to_exclude)
            
def replace_linear_with_target_and_quantize(module, 
                               target_class, module_name_to_exclude):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not \
        any([x == name for x in module_name_to_exclude]):
            old_bias = child.bias
            old_weight = child.weight

            new_module = target_class(child.in_features, 
                                      child.out_features, 
                                      old_bias is not None, 
                                      child.weight.dtype)
            setattr(module, name, new_module)

            getattr(module, name).quantize(old_weight)
            
            if old_bias is not None:
              getattr(module, name).bias = old_bias
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target_and_quantize(child, 
                     target_class, module_name_to_exclude)