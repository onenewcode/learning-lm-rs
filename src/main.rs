use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
};
mod cache;
mod chat;
mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;
use cache::CACHE_MANGER;
use clap::{Args, Parser, Subcommand};
use model::Llama;
use tokenizers::Tokenizer;
static MY_LLAMA: OnceLock<Arc<Llama<f32>>> = OnceLock::new();
static MY_TOKENIZER: OnceLock<Arc<Tokenizer>> = OnceLock::new();
fn main() {
    // 加载模型
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    // 初始化一些全局变量
    let _ = MY_LLAMA.set(Arc::new(model::Llama::<f32>::from_safetensors(&model_dir)));
    let _ = MY_TOKENIZER.set(Arc::new(
        Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap(),
    ));
    let _ = unsafe { CACHE_MANGER.set(HashMap::new()) };
    match Cli::parse().command {
        Commands::Chat(a) => {
            use chat::server::cmd_server;
            if a.mode == "cmd" {
                cmd_server();
            }
        }
    }
}
#[derive(Parser)]
#[clap(version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}
#[derive(Subcommand)]
enum Commands {
    Chat(ChatArgs),
}

// 用于映射参数的结构体
#[derive(Args, Default)]
struct ChatArgs {
    #[clap(short, long, default_value_t = 0)]
    user_id: u32,
    /// Session id.
    #[clap(short, long, default_value_t = 0)]
    session_id: u32,
    /// Model directory.
    #[clap(short, long, default_value = "cmd")]
    mode: String,
}
