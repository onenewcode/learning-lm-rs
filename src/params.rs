use core::slice;
use std::alloc::LayoutErr;

use crate::tensor::Tensor;
use crate::{config::LlamaConfigJson, operators::MyFloat};
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

impl<T> LLamaParams<T>
where
    T: MyFloat,
{
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let layers = config.num_hidden_layers;
        safetensor.names().iter().for_each(|name| {
            println!("{}", name);
        });
        let get_tensor = |name: &str| {
            let tensor_view = safetensor.tensor(name).expect("Failed to get tensor");
            let mut data = vec![];
            for chunk in tensor_view.data().chunks_exact(T::t_size()) {
                let bytes = chunk.try_into().expect("slice with incorrect length");
                let f = T::from_myu8(bytes);
                data.push(f);
            }
            Tensor::new(data, &tensor_view.shape().to_vec())
        };
        Self {
            embedding_table: get_tensor("model.embed_tokens.weight"),
            rms_att_w: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.input_layernorm.weight")))
                .collect(),
            wq: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.q_proj.weight")))
                .collect(),
            wk: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.k_proj.weight")))
                .collect(),
            wv: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.v_proj.weight")))
                .collect(),
            wo: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.o_proj.weight")))
                .collect(),
            rms_ffn_w: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.post_attention_layernorm.weight")))
                .collect(),
            w_up: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.up_proj.weight")))
                .collect(),
            w_gate: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.gate_proj.weight")))
                .collect(),
            w_down: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.down_proj.weight")))
                .collect(),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
