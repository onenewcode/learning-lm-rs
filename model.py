import torch
import os
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
# from bitsandbytes import 

# quantization_config = QuantizationConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# 模型路径
model_id = './models/story/'

# 模型加载，转换
model = AutoModelForCausalLM.from_pretrained(
    model_id, # 模型路径
    local_files_only=True,# 仅从本地加载模型，不从网络下载。
    torch_dtype=torch.float16, # 型权重的数据类型转换为 float16
     quantization_config = BitsAndBytesConfig(
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ),
    # device_map='auto' # 自动分配设备
)


output = "./soulteary/Chinese-Llama-2-7b-4bit"
if not os.path.exists(output):
    os.mkdir(output)

model.save_pretrained(output)
print("done")

