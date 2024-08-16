from transformers import LlamaForCausalLM,LlamaTokenizer
def generate(parameters):
    

model_path = "/models/story/"

# 加载预训练的模型和相应的分词器
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = LlamaForCausalLM.from_pretrained(
    model_path,
    # device_map="auto",  # 如果有GPU，这会自动将模型分配到GPU上
)
tokenizer = LlamaTokenizer.from_pretrained(model_path)
input = "Once upon a time"

