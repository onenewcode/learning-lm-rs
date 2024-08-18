mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;
mod chat;
use std::{path::PathBuf, sync::Arc};
use tokenizers::Tokenizer;
use chat::chat::Chat;
// todo 完成chat
fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama =Arc::new(model::Llama::<f32>::from_safetensors(&model_dir));
    let tokenizer = Arc::new(Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap());
    let input = "system";
    let mut c= Chat::start_chat(0,llama,tokenizer.clone(),20);
    let output_ids = c.chat_generate(input);
    print!("{}", tokenizer.decode(&output_ids, true).unwrap());
}
