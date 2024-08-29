
use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators::{
    self as OP, masked_softmax, matmul_transb, rms_norm, silu, vec_multi, vec_multi_wight,
};
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
use tokio::sync::mpsc::UnboundedSender;
pub struct Llama<T> {
    vocab: usize,    // vocab size
    n_layers: usize, // number of layers
    n_q_h: usize,    // number of heads for q
    n_kv_h: usize,   // number of heads for k and v
    /// 隐藏状态的维度
    d: usize, // dimension of hidden states
    dqkv: usize,     // length of a single q, k, or v vector
    /// mlp中间状态的维度
    di: usize, // dimension of intermediate states
    eps: f16,        // epsilon for RMS normalization
    rope_theta: f16, // rope theta for rope initialization
    max_seq_len: usize, // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32, // start token id
    eos_token_id: u32, // end token id
}

impl Llama<f16> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);
        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps:  0.000001,
            rope_theta:10000.0,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f16> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }
    // 推理单元重复使用
    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f16>) -> Tensor<f16> {
        // 输入文本的长度
        let seq_len = input.size();
        let past_seq_len = cache.len();
        // 更新已经处理过的文本的长度
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        // 用于判断多少个q，对应一个kv
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f16>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f16>::default(&vec![seq_len, self.d]);
        // shape 6*128
        let mut q_buf = Tensor::<f16>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        // shape 4*2*6*6
        let mut att_scores =
            Tensor::<f16>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f16>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f16>::default(&vec![seq_len, self.di]);

        // Computation Starts Here 进行推理
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table); // 获取文本的词向量

        for layer in 0..self.n_layers {
            // 归一化
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );
            // 初始化qkv，kv从cache中获取
            // shape 6*128
            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
                                                                                  // shape 6*64
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
                                                             // 这里我们的wq的矩阵和
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            //  每个词向量对应的查询矩阵
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
                                                       // self_attention
            {
                // score = Q @ K.T / sqrt(dim)
                q.reshape(&vec![seq_len, self.n_q_h * self.dqkv]);
                // 获得的形状，应该为n_q_h * seq_len * total
                vec_multi(
                    &mut att_scores,
                    q,
                    full_k,
                    1. / (self.dqkv as f16).sqrt(),
                    true,
                );
                masked_softmax(&mut att_scores);
                // x = attn @ V
                // 这里需要用到权重乘法，即上一步得出的是权重，接下来每一个权重对应的V向量
                // vec_multi(&mut hidden_states, att_scores, &full_k, 1., false);
                vec_multi_wight(&mut hidden_states, &att_scores, &full_v);
                // x shape 6*128,
                matmul_transb(
                    &mut residual,
                    1.,
                    &hidden_states,
                    &self.params.wo[layer],
                    1.,
                )
            }

            // todo!("down_proj matmul and add residual");
            // todo!("mlp(...)");
            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f16>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }

    pub fn generate(
        &self,
        kvcache: &mut KVCache<f16>, //kv缓存
        token_ids: &[u32],
        top_p: f16,
        top_k: u32,
        temperature: f16,
        sender: UnboundedSender<u32>,
    ) -> Vec<u32> {
        let mut result = Vec::<u32>::new();
        let input = Tensor::<u32>::new(Vec::from(token_ids), &vec![token_ids.len()]);
        // 推理用户输入的信息
        let mut tmp = Tensor::<u32>::new(
            vec![OP::random_sample(
                &self.forward(&input, kvcache),
                top_p,
                top_k,
                temperature,
            )],
            &vec![1],
        );
        // 获取临时数据

        result.push(tmp.data()[0]);
        while kvcache.len() < self.max_seq_len && tmp.data()[0] != self.eos_token_id {
            match sender.send(tmp.data()[0]) {
                Ok(_) => {}
                Err(v) => {
                    println!("{:?}", v);
                }
            }
            // 更新最新推理的数据
            let tt = OP::random_sample(&self.forward(&tmp, kvcache), top_p, top_k, temperature);
            result.push(tt);
            unsafe {
                tmp.data_mut()[0] = tt;
            }
        }
        let _ = sender.clone();
        result
    }
}
fn mlp(
    residual: &mut Tensor<f16>,      // 4 2
    hidden_states: &mut Tensor<f16>, //4 2
    gate: &mut Tensor<f16>,          //4 3
    up: &mut Tensor<f16>,            // 4,3
    w_up: &Tensor<f16>,              // 3,2
    w_down: &Tensor<f16>,            // 2.3
    w_gate: &Tensor<f16>,            // 3,2
    rms_w: &Tensor<f16>,             // 2
    eps: f16,
) {
    rms_norm(hidden_states, residual, rms_w, eps);
    matmul_transb(gate, 0., hidden_states, w_gate, 1.0);
    matmul_transb(up, 0., hidden_states, w_up, 1.0);
    //  hidden = gate * sigmoid(gate) * up ## silu
    silu(up, &gate);
    matmul_transb(residual, 1., &up, w_down, 1.);
}
