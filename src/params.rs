use core::slice;
use std::alloc::LayoutErr;

use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::{SafeTensors, View};
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        safetensor.names().iter().for_each(|name| {
            println!("{}", name);
        });
        let get_tensor= |name: &str| { 
            match safetensor.tensor(name)  {
                Ok(data) => {
                    let p:usize=data.shape().iter().product();
                    // 获取引用，只目前只转换成f32类型
                   let new_data=unsafe { slice::from_raw_parts(data.data().as_ptr() as *const f32, p)};
                   // 生成新对象
                    Tensor::new(Vec::from(new_data), &data.shape().to_vec())
                } ,
                // todo
                Err(err) => {
                Tensor::default(&Vec::new())
                },
            }
        };
        
        LLamaParams {
            embedding_table: todo!(),
            rms_att_w: todo!(),
            wq: todo!(),
            wk: todo!(),
            wv: todo!(),
            wo: todo!(),
            rms_ffn_w: todo!(),
            w_up: todo!(),
            w_gate: todo!(),
            w_down: todo!(),
            rms_out_w: todo!(),
            lm_head: todo!(),
        }
    }
}
