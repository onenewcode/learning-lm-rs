use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use crate::{cache::Cache, model::Llama, operators as OP, tensor::Tensor};
// 固定模板
const RENDER: &str="<|im_start|>";
const ROLE:&str="system\n";
pub struct Chat{
    // 对话id
    id:u32,
    // 管理对话历史 todo
    // 模型参数
    model:Arc<Llama<f32>>,
    tokenizer:Arc<Tokenizer>,
    // 缓存 RefCell
    cache:Arc<Mutex<Cache>>,
    // 最大长度
}
impl  Chat  {
    pub fn new(id:u32,model:Arc<Llama<f32>>,cache:Arc<Mutex<Cache>>,tokenizer:Arc<Tokenizer>)->Chat {
        // 判断是否加载以前的对话 todo
        // 如果有对话历史，加载以前的对话
        Chat { id, model:model.clone(), cache: cache,tokenizer: tokenizer}
    }
   pub  fn start_generate(&self,input:&str)->String{
        // 判断是否为空
        let binding = self.tokenizer.encode( format!("{}{}{}", RENDER,ROLE,input).as_str(), true).unwrap();
        let input_ids = binding.get_ids();
       self.tokenizer.decode(&self.generate(input_ids), true).unwrap()
    }
    fn generate(&self,input:&[u32])->Vec<u32>{
        let (top_p, top_k, temperature) = (0.9, 1, 1.);
        let mut kv=self.cache.lock().unwrap();
        // 添加输入信息，输入步长
        kv.append_info(input);
       let v=self.model.generate(kv.get_mut_kvcache(),input,top_p,top_k,temperature);
       kv.append_info(&v);
       v
    }
    pub  fn chat_rollback(&self)->Vec<u32>{
        let mut kv=self.cache.lock().unwrap();
        let input=kv.rollback();
        let (top_p, top_k, temperature) = (0.9, 1, 1.);
        let v=self.model.generate(kv.get_mut_kvcache(),&[input],top_p,top_k,temperature);
        kv.append_info(&v);
        v
    }
    pub fn decode(&self,input:&[u32])->String{
        self.tokenizer.decode(input, true).unwrap()
    }
}