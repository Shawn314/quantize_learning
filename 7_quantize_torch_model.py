from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from helper import W8A16LinearLayer, replace_linear_with_target_and_quantize
import torch
import torch.nn as nn
import torch.nn.functional as F

model_id = "Salesforce/codegen-350M-mono"

model = AutoModelForCausalLM.from_pretrained(model_id, 
                                    torch_dtype=torch.bfloat16, 
                                             low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

print(pipe("def hello_world():", max_new_tokens=20, do_sample=False))

print("Model before:\n\n", model)

replace_linear_with_target_and_quantize(model, 
                                        W8A16LinearLayer, ["lm_head"])
print(pipe.model)
print(pipe("def hello_world():", max_new_tokens=20, 
           do_sample=False)[0]["generated_text"])