use std::{path::PathBuf, sync::{Arc, Mutex}};
mod chat;
mod kvcache;
mod model;
mod operators;
mod params;
mod config;
mod tensor;
mod cache;
use clap::{Args, Parser, Subcommand};
use tokenizers::Tokenizer;


 fn main() {
    // 加载模型
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama =Arc::new(model::Llama::<f32>::from_safetensors(&model_dir));
    let tokenizer = Arc::new(Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap());
    match Cli::parse().command {
        Commands::Chat(a) => {
            use chat::server::cmd_server;
           if a.mode == "cmd" {
            let cache= Arc::new(Mutex::new(cache::Cache::new(llama.new_cache())));
            // 生成对话模型
            let chat = chat::chat::Chat::new(a.session_id, llama, cache,tokenizer);
            cmd_server(chat);
            //   tokio::spawn(cmd_server());
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
