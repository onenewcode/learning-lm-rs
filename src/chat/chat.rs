use std::sync::Arc;
use tokio::sync::oneshot;
use tokenizers::Tokenizer;
use crate::{kvcache::KVCache, model::Llama, operators as OP, tensor::Tensor};
// 固定模板
const RENDER: &str="<|im_start|>";
pub struct Chat{
    // 对话id
    id:u32,
    // 管理对话历史 todo
    // 模型参数
    model:Arc<Llama<f32>>,
    tokenizer:Arc<Tokenizer>,
    // 缓存 RefCell
    cache:KVCache<f32>,
    // 最大长度
    max_len:usize,
    // 用于，接受命令行参数
    input_chan:oneshot::Receiver<String>,
    // 用于输出每次推理的值
    output_chan:oneshot::Sender<String>,
}
impl  Chat  {
    pub fn start_chat(id:u32,model:Arc<Llama<f32>>,tokenizer:Arc<Tokenizer>,max_len:usize)->Chat {
        // 判断是否加载以前的对话 todo
        // 如果有对话历史，加载以前的对话
        let (output_chan,input_chan ) = oneshot::channel::<String>();
        Chat { id, model:model.clone(), cache: model.new_cache(),tokenizer: tokenizer,max_len,input_chan,output_chan}
    }
   pub  fn chat_generate(&mut self,input:&str)->Vec<u32>{
        // 判断是否为空
        let binding = self.tokenizer.encode( format!("{}{}", RENDER, input).as_str(), true).unwrap();
        let input_ids = binding.get_ids();
        let (top_p, top_k, temperature) = (0.9, 4, 1.);
        self.model.generate(&mut self.cache,input_ids,self.max_len,top_p,top_k,temperature,32000)
    }
}