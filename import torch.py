import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import safetensors.torch
# 使用中文版
model_id = "./models/chat"
# 或者，使用原版
# model_id = 'meta-llama/Llama-2-7b-chat-hf'

model = AutoModelForCausalLM.from_pretrained(
    model_id
)
model =  AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 将模型转换为 f16（半精度）
model = model.half()

# 保存转换后的模型
output_dir = model_id+"_f16"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
