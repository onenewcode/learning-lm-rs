use std::{cell::RefCell, collections::HashMap, sync::{Arc, Mutex, OnceLock}};

use crate::kvcache::KVCache;

static CACHE_MANGER: OnceLock<CManger> = OnceLock::new();
type CManger=HashMap<String,Arc<Mutex<Cache>>>;
pub struct Cache{
    kv_cache:KVCache<f32>,
}
impl Cache {
    pub fn get_mut_kvcache(&mut self)->&mut KVCache<f32>{
        &mut self.kv_cache
    }
    pub fn new(kv_cache:KVCache<f32>)->Self{
        Self{kv_cache}
    }
}