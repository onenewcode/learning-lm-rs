use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use crate::{kvcache::KVCache, MY_LLAMA};

pub(crate) static mut CACHE_MANGER: OnceLock<CManger> = OnceLock::new();
type CManger = HashMap<String, Arc<Mutex<Cache>>>;
pub struct Cache {
    kv_cache: KVCache<f32>,
    // 用于记录推理步长
    step: Vec<usize>,
    // 用于记录推理信息
    info: Vec<u32>,
}

impl Cache {
    pub fn get_mut_kvcache(&mut self) -> &mut KVCache<f32> {
        &mut self.kv_cache
    }
    pub fn new(kv_cache: KVCache<f32>) -> Self {
        Self {
            kv_cache,
            step: Vec::new(),
            info: Vec::new(),
        }
    }
    pub fn new_cmanger() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Cache::new(MY_LLAMA.get().unwrap().new_cache())))
    }
    // 用于回滚，同时回滚kvc
    pub fn pop_step(&mut self) {
        self.step.pop();
        self.kv_cache.reset(*self.step.last().unwrap_or(&0));
    }
    pub fn push_step(&mut self, step: usize) {
        self.step.push(step);
    }
    pub fn kvc_len(&self) -> usize {
        self.kv_cache.len()
    }
    pub fn get_info(&self) -> &Vec<u32> {
        &self.info
    }
    pub fn append_info(&mut self, info: &[u32]) {
        // 获取不到置为0
        let last = self.step.last().unwrap_or(&0);
        // 更新步长
        self.step.push(last + info.len());
        // 追加元素
        self.info.extend(info.iter());
    }
    // 回滚，返回推理的最后一个元素
    pub fn rollback(&mut self) -> u32 {
        // 弹出元素
        self.step.pop();
        let last = self.step.last().unwrap();
        self.kv_cache.reset(*last - 1);
        // 清空info不需要的中的元素
        self.info.truncate(*last);
        *self.info.last().unwrap()
    }
}
