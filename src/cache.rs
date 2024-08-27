use crate::{kvcache::KVCache, operators::MyFloat, MY_LLAMA_F16, MY_LLAMA_F32};
use half::f16;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

pub(crate) static mut CACHE_MANGER_F32: OnceLock<CManger<f32>> = OnceLock::new();
pub(crate) static mut CACHE_MANGER_F16: OnceLock<CManger<f16>> = OnceLock::new();
type CManger<T:MyFloat> = HashMap<String, Arc<Mutex<Cache<T>>>>;
// impl ShutDownCallback for CManger {
//     // todo 持久化
//     fn shut_down_callback(&self) {
//         // self.iter().for_each(|(i, cache)| {
//         //     // 序列化Person到JSON字符串
//         //     let json_str = serde_json::to_string_pretty(cache)?;

//         //     // 将序列化后的JSON字符串写入文件
//         //     let mut file = File::create("person.json")?;
//         //     file.write_all(json_str.as_bytes())?;
//         // });
//     }
// }
pub struct Cache<Storage: Default + Copy> {
    kv_cache: KVCache<Storage>,
    // 用于记录推理步长
    step: Vec<usize>,
    // 用于记录推理信息
    info: Vec<u32>,
}

impl<Storage: Default + Copy> Cache<Storage> {
    pub fn get_mut_kvcache(&mut self) -> &mut KVCache<Storage> {
        &mut self.kv_cache
    }
    pub fn new(kv_cache: KVCache<Storage>) -> Self {
        Self {
            kv_cache,
            step: Vec::new(),
            info: Vec::new(),
        }
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
impl Cache<f32> {
    pub fn new_cmanger() -> Arc<Mutex<Self>> {
        // Arc::new(Mutex::new(Cache::new(MY_LLAMA.get().unwrap().new_cache())))
        Arc::new(Mutex::new(Self {
            kv_cache: MY_LLAMA_F32.get().unwrap().new_cache(),
            step: Vec::new(),
            info: Vec::new(),
        }))
    }
}
impl Cache<f16> {
    pub fn new_cmanger() -> Arc<Mutex<Self>> {
        // Arc::new(Mutex::new(Cache::new(MY_LLAMA.get().unwrap().new_cache())))
        Arc::new(Mutex::new(Self {
            kv_cache: MY_LLAMA_F16.get().unwrap().new_cache(),
            step: Vec::new(),
            info: Vec::new(),
        }))
    }
}
