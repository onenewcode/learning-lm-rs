use core::slice;
use crate::tensor::Tensor;
use crate::{config::LlamaConfigJson};
use safetensors::{SafeTensors};
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

impl LLamaParams<f16>
{
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let layers = config.num_hidden_layers;
        safetensor.names().iter().for_each(|name| {
            println!("{}", name);
        });
        let get_tensor = |name: &str| {
            let tensor_view = safetensor.tensor(name).expect("Failed to get tensor");
            let l:usize=tensor_view.shape().iter().product();
            let data=unsafe {
                slice::from_raw_parts(tensor_view.data().as_ptr() as *const f16, l)
            };
           Tensor::new(Vec::from(data), &tensor_view.shape().to_vec())
        };
        Self {
            embedding_table: get_tensor("model.embed_tokens.weight"),
            rms_att_w: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.input_layernorm.weight")))
                .collect(),
            wq: (0..layers)
                .map(|i|{
                    let t=&format!("model.layers.{i}.self_attn.qkv_proj.weight");
                    match safetensor.tensor(t) {
                        Ok(data) => {
                            // 获取wq长度
                            let start=2*0;
                            let end: usize =2*(start+ config.hidden_size*config.hidden_size);
                            let mut tdata = vec![];
                            for chunk in data.data()[start..end].chunks_exact(2) {
                                let bytes: [u8; 2] = chunk.try_into().expect("slice with incorrect length");
                                let f = f16::from_le_bytes(bytes);
                                tdata.push(f);
                            }
                            Tensor::new(tdata, &vec![config.hidden_size,config.hidden_size])
                        }
                        Err(err) => panic!("加载模型失败{:?}", err),
                    }
                }).collect(),
            wk: (0..layers)
                .map(|i|{
                    let t=&format!("model.layers.{i}.self_attn.qkv_proj.weight");
                    match safetensor.tensor(t) {
                        Ok(data) => {
                        let start=2*(config.hidden_size*config.hidden_size);
                        let end: usize = start+2*(config.hidden_size*config.hidden_size*config.num_key_value_heads/config.num_attention_heads);
                        let mut tdata = vec![];
                        for chunk in data.data()[start..end].chunks_exact(2) {
                            let bytes: [u8; 2] = chunk.try_into().expect("slice with incorrect length");
                            let f = f16::from_le_bytes(bytes);
                            tdata.push(f);
                        }
                            Tensor::new(tdata, &vec![config.hidden_size*config.num_key_value_heads/config.num_attention_heads,config.hidden_size])
                        }
                        Err(err) => panic!("加载模型失败{:?}", err),
                    }
                } )
                .collect(),
            wv: (0..layers)
                .map(|i| {
                    let t=&format!("model.layers.{i}.self_attn.qkv_proj.weight");
                    match safetensor.tensor(t) {
                        Ok(data) => {
                            let start= 2*(config.hidden_size*(config.hidden_size+(config.hidden_size*config.num_key_value_heads/config.num_attention_heads)));
                            let end=start+2*(config.hidden_size*config.hidden_size*config.num_key_value_heads/config.num_attention_heads);
                            let mut tdata = vec![];
                            for chunk in data.data()[start..end].chunks_exact(2) {
                                let bytes: [u8; 2] = chunk.try_into().expect("slice with incorrect length");
                                let f = f16::from_le_bytes(bytes);
                                tdata.push(f);
                            }
                            let my= Tensor::new(tdata, &vec![config.hidden_size*config.num_key_value_heads/config.num_attention_heads,config.hidden_size]);
                            my
                        }
                        Err(err) => panic!("加载模型失败{:?}", err),
                    }
                } )
                .collect(),
            wo: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.o_proj.weight")))
                .collect(),
            rms_ffn_w: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.post_attention_layernorm.weight")))
                .collect(),
            w_gate: (0..layers)
                .map(|i|{
                    let t=&format!("model.layers.{i}.mlp.gate_up_proj.weight");
                    match safetensor.tensor(t) {
                        Ok(data) => {
                            let start=2*0;
                            let end: usize =2*config.intermediate_size*config.hidden_size;
                            // 获取引用，只目前只转换成f32类型
                            let mut tdata = vec![];
                            for chunk in data.data()[start..end].chunks_exact(2) {
                                let bytes: [u8; 2] = chunk.try_into().expect("slice with incorrect length");
                                let f = f16::from_le_bytes(bytes);
                                tdata.push(f);
                            }
                            // 生成新对象
                            Tensor::new(tdata,&vec![config.intermediate_size,config.hidden_size])
                        }
                        Err(err) => panic!("加载模型失败{:?}", err),
                    }
                } )
                .collect(),
                w_up: (0..layers)
                .map(|i|{
                    let t=&format!("model.layers.{i}.mlp.gate_up_proj.weight");
                    match safetensor.tensor(t) {
                        Ok(data) => {
                            let start=2*(config.intermediate_size*config.hidden_size);
                            let end: usize =start+ 2*(config.intermediate_size*config.hidden_size);
                            // 获取引用，只目前只转换成f32类型
                            let mut tdata = vec![];
                            for chunk in data.data()[start..end].chunks_exact(2) {
                                let bytes: [u8; 2] = chunk.try_into().expect("slice with incorrect length");
                                let f = f16::from_le_bytes(bytes);
                                tdata.push(f);
                            }
                            // 生成新对象
                            Tensor::new(tdata, &vec![config.intermediate_size,config.hidden_size])
                        }
                        Err(err) => panic!("加载模型失败{:?}", err),
                    }
                } )
                .collect(),
            w_down: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.down_proj.weight")))
                .collect(),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
