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
use half::f16;
static MY_LLAMA_F32: OnceLock<Arc<Llama<f32>>> = OnceLock::new();
static MY_LLAMA_F16: OnceLock<Arc<Llama<f16>>> = OnceLock::new();
static MY_TOKENIZER: OnceLock<Arc<Tokenizer>> = OnceLock::new();
trait ShutDownCallback {
    fn shut_down_callback(&self);
}
#[tokio::main]
async fn main() {
    // 加载模型
 
    match Cli::parse().command {
        Commands::Chat(a) => {
            use chat::server::cmd_server;
            if a.mode == "cmd" {
                let model_dir = PathBuf::from(a.model);
                let _ = MY_TOKENIZER.set(Arc::new(
                    Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap(),
                ));
                // 初始化一些全局变量
                let _ = MY_LLAMA_F32.set(Arc::new(model::Llama::from_safetensors(&model_dir)));
                // todo f16 代码
                // let _ = MY_LLAMA_F16.set(Arc::new(model::Llama::from_safetensors(&model_dir)));
             
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
    #[clap(long, default_value = "cmd")]
    mode: String,
    #[clap(short,long, required=true)]
    model: String,
    
}
#[macro_export]
macro_rules! print_now {
    ($($arg:tt)*) => {{
        use std::io::Write;

        print!($($arg)*);
        std::io::stdout().flush().unwrap();
    }};
}
