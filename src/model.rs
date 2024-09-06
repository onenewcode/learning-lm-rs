use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators::{
    self as OP, masked_softmax, matmul_transb, rms_norm, silu, vec_multi, vec_multi_wight, MyFloat,
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
    eps: f32,        // epsilon for RMS normalization
    rope_theta: f32, // rope theta for rope initialization
    max_seq_len: usize, // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32, // start token id
    eos_token_id: u32, // end token id
}
impl<T> Llama<T>
where
    T: MyFloat,
{
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
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<T> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }
    // 推理单元重复使用
    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<T>) -> Tensor<T> {
        // 输入文本的长度
        let seq_len = input.size();
        let past_seq_len = cache.len();
        // 更新已经处理过的文本的长度
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        // 用于判断多少个q，对应一个kv
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<T>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<T>::default(&vec![seq_len, self.d]);
        // shape 6*128
        let mut q_buf = Tensor::<T>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        // shape 4*2*6*6
        let mut att_scores =
            Tensor::<T>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<T>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<T>::default(&vec![seq_len, self.di]);

        // Computation Starts Here 进行推理
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table); // 获取文本的词向量

        for layer in 0..self.n_layers {
            // 归一化
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                T::from_f32(self.eps).unwrap(),
            );
            // 初始化qkv，kv从cache中获取
            // shape 6*128
            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
                                                                                  // shape 6*64
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
                                                             // 这里我们的wq的矩阵和
            OP::matmul_transb(
                q,
                T::zero(),
                &hidden_states,
                &self.params.wq[layer],
                T::one(),
            );
            OP::matmul_transb(
                k,
                T::zero(),
                &hidden_states,
                &self.params.wk[layer],
                T::one(),
            );
            OP::matmul_transb(
                v,
                T::zero(),
                &hidden_states,
                &self.params.wv[layer],
                T::one(),
            );
            //  每个词向量对应的查询矩阵
            OP::rope::<T>(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                T::from_f32(self.rope_theta).unwrap(),
            );
            OP::rope::<T>(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                T::from_f32(self.rope_theta).unwrap(),
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
                    T::from_f32(1. / (self.dqkv as f32).sqrt()).unwrap(),
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
                    T::one(),
                    &hidden_states,
                    &self.params.wo[layer],
                    T::one(),
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
                T::from_f32(self.eps).unwrap(),
            );
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<T>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            T::from_f32(self.eps).unwrap(),
        );

        OP::matmul_transb(
            &mut logits,
            T::zero(),
            &hidden_states,
            &self.params.lm_head,
            T::one(),
        );

        logits
    }

    pub fn generate(
        &self,
        kvcache: &mut KVCache<T>, //kv缓存
        token_ids: &[u32],
        top_p: f32,
        top_k: u32,
        temperature: f32,
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

// fn self_attention(
//     hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
//     att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
//     q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)  (seq_len, q_head×dim)
//     k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
//     v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
//     n_kv_h: usize,
//     n_groups: usize,
//     seq_len: usize,
//     total_seq_len: usize,
//     dqkv: usize,
// ) {
// }

fn mlp<T>(
    residual: &mut Tensor<T>,      // 4 2
    hidden_states: &mut Tensor<T>, //4 2
    gate: &mut Tensor<T>,          //4 3
    up: &mut Tensor<T>,            // 4,3
    w_up: &Tensor<T>,              // 3,2
    w_down: &Tensor<T>,            // 2.3
    w_gate: &Tensor<T>,            // 3,2
    rms_w: &Tensor<T>,             // 2
    eps: T,
) where
    T: MyFloat,
{
    rms_norm(hidden_states, residual, rms_w, eps);
    matmul_transb(gate, T::zero(), hidden_states, w_gate, T::one());
    matmul_transb(up, T::zero(), hidden_states, w_up, T::one());
    //  hidden = gate * sigmoid(gate) * up ## silu
    silu(up, &gate);
    matmul_transb(residual, T::one(), &up, w_down, T::one());
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );
    residual.print();
    residual.print();
    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use crate::tensor::float_eq;
    use std::path::PathBuf;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(
        &model.params.embedding_table.data()[50],
        &0.14453125,
        1e-6
    ));
    assert_eq!(
        model.params.lm_head.data()[10],
        model.params.embedding_table.data()[10]
    );
    assert!(float_eq(
        &model.params.rms_att_w[0].data()[10],
        &0.18652344,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_ffn_w[1].data()[10],
        &0.32421875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_out_w.data()[100],
        &0.73046875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.w_down[0].data()[100],
        &-0.0625,
        1e-6
    ));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(
        &model.params.w_gate[1].data()[100],
        &0.296875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wq[1].data()[100],
        &0.032226563,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wk[1].data()[100],
        &-0.21386719,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wv[0].data()[100],
        &0.041015625,
        1e-6
    ));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));
}
