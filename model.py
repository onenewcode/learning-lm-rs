from transformers import LlamaForCausalLM
model_name_or_path = "./models/story/"

# 加载预训练的模型和相应的分词器
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = LlamaForCausalLM.from_pretrained(
    model_name_or_path, # 如果有GPU，这会自动将模型分配到GPU上
    trust_remote_code=True  # 允许加载模型作者提供的额外代码
)