use std::vec;

use crate::tensor::Tensor;
/// 获取编码后，每个词代表的向量，组成一个矩阵
// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f16>, indices: &Tensor<u32>, table: &Tensor<f16>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f16>, start_pos: usize, theta: f16) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f16 / theta.powf((i * 2) as f16 / d as f16);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f16>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let mut sum=0.; 
            (0..boundary)
                .for_each(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    sum+=e;
                });
            // let sum = (0..boundary)
            // .map(|j| {
            //     let e = (data[offset + j] - max).exp();
            //     data[offset + j] = e;
            //     e
            // })
            // .sum::<f32>();
            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f16>, x: &Tensor<f16>, w: &Tensor<f16>, epsilon: f16) {
    let y_data = unsafe { y.data_mut() };

    let mut shape = vec![];
    // 判断是否是一维
    if x.shape().len() == 1 {
        shape.push(1);
        shape.push(x.shape()[0]);
    } else {
        shape = x.shape().clone();
    }
    for i in 0..shape[0] {
        let row_range = i * shape[1]..(i + 1) * shape[1];
        let mut sum=0.;
        x.data()[row_range.clone()]
            .iter()
            .for_each(|&x| {
                sum+=x.powi(2);
            });
        let sq = ((sum
            / shape[1] as f16)
            + epsilon).sqrt();
        y_data[row_range.clone()]
            .iter_mut()
            .zip(x.data()[row_range].iter().zip(w.data().iter()))
            .for_each(|(y_d, (x_d, w_d))| {
                *y_d = (*w_d * *x_d) / sq;
            });
    }
}

// y = sigmoid(x) * x * y
// hint: this is an element-wise operation
pub fn silu(y: &mut Tensor<f16>, x: &Tensor<f16>) {
    let len = y.size();
    assert!(len == x.size());

    let y_data = unsafe { y.data_mut() };
    let x_data = x.data();
    for i in 0..x_data.len() {
        y_data[i] = y_data[i] * x_data[i] / (1.0 + (-x_data[i]).exp())
    }
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f16>, beta: f16, a: &Tensor<f16>, b: &Tensor<f16>, alpha: f16) {
    assert!(
        b.shape().len() == 2,
        "matmul_transb of dimensions must be at least 2"
    );
    assert!(
        a.shape().len() == 2,
        "matmul_transb of dimensions must be at least 2"
    );
    if a.shape()[1] != b.shape()[1] {
        panic!("matmul_transb of 大小不相同");
    }
    // 计算 num 是 multiple 的几倍
    // 默认当二维数组处理
    let shape = vec![a.shape()[0], b.shape()[0]];
    let mid = a.shape()[1];
    let c_data = unsafe { c.data_mut() };
    let mut offset = 0;

    for i in 0..shape[0] {
        let row = &a.data()[i * mid..(i + 1) * mid];
        for j in 0..shape[1] {
            let column = &b.data()[j * mid..(j + 1) * mid];
            let mut sum=0.;
            row.iter().zip(column).for_each(|(a, b)| sum+=a * b);
            c_data[offset] = alpha * sum + c_data[offset] * beta;
            offset += 1;
        }
    }
}
// 向量乘法
// t判断是否需要转置,alpha,参数暂未使用
pub fn vec_multi(c: &mut Tensor<f16>, a: &Tensor<f16>, b: &Tensor<f16>, alpha: f16, t: bool) {
    // 判断c，长度是否大于二
    assert!(
        c.shape().len() > 2,
        "vec_multi of dimensions must be at least 2"
    );
    // a 重要，用于切分数据
    assert!(a.shape().len() == 2, "vec_multi of dimensions must be 2");
    assert!(b.shape().len() == 2, "vec_multi of dimensions must be 2");
    let shape = c.shape();
    // 获取矩阵的行列数
    let (row, column) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    // 获取n_q_h，用于分组
    let q_head_len = shape[..shape.len() - 2].iter().product::<usize>();
    // 确定qk的倍数对应关系
    let q_k_reflect = a.shape()[1] / b.shape()[1];
    let vec_len = a.shape()[1] / q_head_len;
    let a_data = a.data();
    // 用于获取q_head需要进行跳过的数值
    let a_skip = a.shape()[1];
    let b_data = b.data();
    // 用于获取k_head需要进行跳过的数值
    let b_skip = b.shape()[1];
    let data = unsafe { c.data_mut() };
    // 清理脏数据
    data.fill(0.);
    let mut c_data_offset = 0;
    if t {
        // 用于分组计算，每个输入，在每个请求头下的vjiv
        for i in 0..q_head_len {
            // 计算一个输入值，在一个请求头下的total中的所有v
            for j in 0..row {
                // 临时q_head 值,j*a_skip用于跳过多头i*16用于跳过单头
                let a_tmp =
                    &a_data[(i * vec_len + j * a_skip)..(i * vec_len + j * a_skip) + vec_len];
                // 计算单一v
                for k in 0..column {
                    let b_tmp = &b_data[(k * b_skip + (i / q_k_reflect) * vec_len)
                        ..(k * b_skip + (i / q_k_reflect) * vec_len) + vec_len];
                    data[c_data_offset] = a_tmp
                        .iter()
                        .zip(b_tmp.iter())
                        .fold(0., |tmp, (a_val, b_val)| tmp + a_val * b_val)
                        * alpha;
                    c_data_offset += 1;
                }
            }
        }
    }
}
// 只用于得分计算
// a代表所处理的权重,b代表所要乘的向量
pub fn vec_multi_wight(c: &mut Tensor<f16>, a: &Tensor<f16>, b: &Tensor<f16>) {
    assert!(
        b.shape().len() == 2,
        "matmul_transb of dimensions must be at least 2"
    );
    assert!(
        a.shape().len() == 4,
        "matmul_transb of dimensions must be  4 是att_scores)"
    );
    let q_header_len = a.shape()[..a.shape().len() - 2].iter().product::<usize>();
    let shape = a.shape();
    // 获取矩阵的行列数
    let (row, column) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    // 获取计算向量的长度
    let vec_len = b.shape()[1] / a.shape()[0];
    // 确认a，b需要的对应关系,默认a的长度大于b的长度
    let n_groups = a.shape()[1];
    let b_column = b.shape()[1];
    let mut data = unsafe { c.data_mut() };
    // 清理脏数据
    data.fill(0.);
    for i in 0..q_header_len {
        // 获取当前q下的的全部注意力
        let a_data = &a.data()[i * row * column..(i + 1) * row * column];
        // 循环计算每个当前q下，每个输入的v权重
        for c_i in 0..row {
            // 用于标记当前计算到那一列
            let mut b_data_row_offset = 0;
            let tmp_c_offset = n_groups * b_column * c_i + i * vec_len;
            // 获取c存储当先向量的位置，
            let tmp_c = &mut data[tmp_c_offset..tmp_c_offset + vec_len];
            // 获取一个输入的全部注意力
            a_data[c_i * column..(c_i + 1) * column]
                .iter()
                .for_each(|tmp| {
                    // 获取q，对应的v b_data_row_offset*b_column表示要跳过的input
                    // (q_header_len/n_groups)*vec_len 表示q对应的v
                    let tmp_offset = b_data_row_offset * b_column + (i / n_groups) * vec_len;
                    let b_data = &b.data()[tmp_offset..tmp_offset + vec_len];
                    b_data.iter().zip(tmp_c.iter_mut()).for_each(|(t_b, t_c)| {
                        *t_c += t_b * tmp;
                    });
                    // 进行偏移
                    b_data_row_offset += 1;
                });
        }
    }
}
// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f16>, top_p: f16, top_k: u32, temperature: f16) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f16,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f16)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f16)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;

    let plimit =rand::random::<f32>() as f16  * f16::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}