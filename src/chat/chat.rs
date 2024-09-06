use crate::{
    cache::Cache, model::Llama, operators::MyFloat, print_now, MY_LLAMA_F16, MY_LLAMA_F32,
    MY_TOKENIZER,
};
use half::f16;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
// 固定模板
const RENDER: &str = "";
const ROLE: &str = "assistant <s>";
pub struct Chat<C: Default + Copy + MyFloat> {
    // 对话id
    id: String,
    // 管理对话历史 todoC
    // 模型参数
    model: Arc<Llama<C>>,
    tokenizer: Arc<Tokenizer>,
    // 缓存 RefCell
    cache: Arc<Mutex<Cache<C>>>,
    // 最大长度
}
impl Chat<f32> {
    pub fn new(
        id: String,
        model: Arc<Llama<f32>>,
        cache: Arc<Mutex<Cache<f32>>>,
        tokenizer: Arc<Tokenizer>,
    ) -> Chat<f32> {
        // 判断是否加载以前的对话 todo
        // 如果有对话历史，加载以前的对话
        Chat {
            id,
            model: model.clone(),
            cache: cache,
            tokenizer: tokenizer,
        }
    }
    pub fn new_chat(id: String, cache: Arc<Mutex<Cache<f32>>>) -> Self {
        Chat {
            id,
            model: MY_LLAMA_F32.get().unwrap().clone(),
            cache: cache,
            tokenizer: MY_TOKENIZER.get().unwrap().clone(),
        }
    }
    pub fn start_generate(self: Arc<Self>, input: &str) -> UnboundedReceiver<u32> {
        // 判断是否为空
        let binding = self
            .tokenizer
            .encode(format!("{}{}{}", RENDER, ROLE, input).as_str(), true)
            .unwrap();
        let (s, r) = unbounded_channel::<u32>();
        tokio::task::spawn_blocking(move || {
            // &Vec::from(binding.get_ids())这里进行深拷贝
            Self::generate(
                &Vec::from(binding.get_ids()),
                self.cache.clone(),
                self.model.clone(),
                s,
            );
        });
        r
    }
    fn generate(
        input: &[u32],
        cache: Arc<Mutex<Cache<f32>>>,
        model: Arc<Llama<f32>>,
        sender: UnboundedSender<u32>,
    ) {
        let (top_p, top_k, temperature) = (0.7, 1, 1.);
        let mut kv = cache.lock().unwrap();
        // 添加输入信息，输入步长
        kv.append_info(input);
        let v = model.generate(
            kv.get_mut_kvcache(),
            input,
            top_p,
            top_k,
            temperature,
            sender,
        );
        kv.append_info(&v);
    }
    pub fn chat_rollback(
        self: Arc<Self>,
        session_len: usize,
    ) -> Result<UnboundedReceiver<u32>, String> {
        // 判断回滚的长度是否超出该会话的长度
        if self.cache.lock().unwrap().get_step_len() < 2 * session_len - 1 {
            return Err("回滚长度超出该会话的长度".to_string());
        }
        let input = {
            let mut kv = self.cache.lock().unwrap();
            kv.rollback(2 * session_len - 1)
        };
        let (s, r) = unbounded_channel::<u32>();
        tokio::task::spawn_blocking(move || {
            Self::generate(&[input], self.cache.clone(), self.model.clone(), s);
        });
        Ok(r)
    }
    pub async fn chat_output(&self, r: &mut UnboundedReceiver<u32>) {
        print_now!("chat id {}\n ", self.chat_id());
        loop {
            match r.recv().await {
                Some(v) => {
                    crate::print_now!("{} ", self.decode(&[v]))
                }
                None => {
                    println!();
                    break;
                }
            }
        }
    }
    pub fn decode(&self, input: &[u32]) -> String {
        self.tokenizer.decode(input, true).unwrap()
    }
    pub fn cache(&self) -> Arc<Mutex<Cache<f32>>> {
        self.cache.clone()
    }
    pub fn chat_id(&self) -> String {
        self.id.clone()
    }
}
impl Chat<f16> {
    pub fn new(
        id: String,
        model: Arc<Llama<f16>>,
        cache: Arc<Mutex<Cache<f16>>>,
        tokenizer: Arc<Tokenizer>,
    ) -> Chat<f16> {
        // 判断是否加载以前的对话 todo
        // 如果有对话历史，加载以前的对话
        Chat {
            id,
            model: model.clone(),
            cache: cache,
            tokenizer: tokenizer,
        }
    }
    pub fn new_chat(id: String, cache: Arc<Mutex<Cache<f16>>>) -> Self {
        Chat {
            id,
            model: MY_LLAMA_F16.get().unwrap().clone(),
            cache: cache,
            tokenizer: MY_TOKENIZER.get().unwrap().clone(),
        }
    }
    pub fn start_generate(self: Arc<Self>, input: &str) -> UnboundedReceiver<u32> {
        // 判断是否为空
        let binding = self
            .tokenizer
            .encode(format!("{}{}{}", RENDER, ROLE, input).as_str(), true)
            .unwrap();
        let (s, r) = unbounded_channel::<u32>();
        tokio::task::spawn_blocking(move || {
            // &Vec::from(binding.get_ids())这里进行深拷贝
            Self::generate(
                &Vec::from(binding.get_ids()),
                self.cache.clone(),
                self.model.clone(),
                s,
            );
        });
        r
    }
    fn generate(
        input: &[u32],
        cache: Arc<Mutex<Cache<f16>>>,
        model: Arc<Llama<f16>>,
        sender: UnboundedSender<u32>,
    ) {
        let (top_p, top_k, temperature) = (0.7, 1, 1.);
        let mut kv = cache.lock().unwrap();
        // 添加输入信息，输入步长
        kv.append_info(input);
        let v = model.generate(
            kv.get_mut_kvcache(),
            input,
            top_p,
            top_k,
            temperature,
            sender,
        );
        kv.append_info(&v);
    }
    pub fn chat_rollback(
        self: Arc<Self>,
        session_len: usize,
    ) -> Result<UnboundedReceiver<u32>, String> {
        // 判断回滚的长度是否超出该会话的长度
        if self.cache.lock().unwrap().get_step_len() < 2 * session_len - 1 {
            return Err("回滚长度超出该会话的长度".to_string());
        }
        let input = {
            let mut kv = self.cache.lock().unwrap();
            kv.rollback( 2 * session_len - 1 )
        };
        let (s, r) = unbounded_channel::<u32>();
        tokio::task::spawn_blocking(move || {
            Self::generate(&[input], self.cache.clone(), self.model.clone(), s);
        });
        Ok(r)
    }
    pub async fn chat_output(&self, r: &mut UnboundedReceiver<u32>) {
        print_now!("chat id {}\n ", self.chat_id());
        loop {
            match r.recv().await {
                Some(v) => {
                    crate::print_now!("{} ", self.decode(&[v]))
                }
                None => {
                    println!();
                    break;
                }
            }
        }
    }
    pub fn decode(&self, input: &[u32]) -> String {
        self.tokenizer.decode(input, true).unwrap()
    }
    pub fn cache(&self) -> Arc<Mutex<Cache<f16>>> {
        self.cache.clone()
    }
    pub fn chat_id(&self) -> String {
        self.id.clone()
    }
}
