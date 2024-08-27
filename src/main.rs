#![feature(once_cell_get_mut)]
use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
};
mod cache;
mod chat;
mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;
use clap::{Args, Parser, Subcommand};
use model::Llama;
use tokenizers::Tokenizer;
static MY_LLAMA: OnceLock<Arc<Llama<f32>>> = OnceLock::new();
static MY_TOKENIZER: OnceLock<Arc<Tokenizer>> = OnceLock::new();
trait ShutDownCallback {
    fn shut_down_callback(&self);
}
#[tokio::main]
async fn main() {
    // 加载模型
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    // 初始化一些全局变量
    let _ = MY_LLAMA.set(Arc::new(model::Llama::<f32>::from_safetensors(&model_dir)));
    let _ = MY_TOKENIZER.set(Arc::new(
        Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap(),
    ));
    match Cli::parse().command {
        Commands::Chat(a) => {
            use chat::server::cmd_server;
            if a.mode == "cmd" {
                cmd_server().await;
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
#[macro_export]
macro_rules! print_now {
    ($($arg:tt)*) => {{
        use std::io::Write;

        print!($($arg)*);
        std::io::stdout().flush().unwrap();
    }};
}
