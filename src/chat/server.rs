use crate::{
    cache::{self, Cache, CACHE_MANGER},
    chat::chat::Chat,
};
use std::{
    collections::HashMap,
    io::{self, BufRead, Write},
    process,
    sync::Arc,
};
enum ChatMessage {
    Chat,
    Rollback,
    Switch(String),
    Exit,
    Error(String),
}
pub async fn cmd_server() {
    let stdin = io::stdin();
    let mut handle = stdin.lock();
    // 初始化默认Chat
    let mut chat = Arc::new(Chat::new_chat("1".to_owned(), cache::Cache::new_cmanger()));
    // 限制作用域
    {
        // 初始化
        let c = unsafe { CACHE_MANGER.get_mut_or_init(|| HashMap::new()) };
        c.insert(chat.chat_id(), chat.cache());
    }
    //  创建一个可安全共享的引用
    loop {
        print!("请输入推理文本 (输入 '>exit' 退出程序): ");
        io::stdout().flush().unwrap(); // 确保输出立即刷新

        let mut input = String::new();
        handle.read_line(&mut input).expect("读取失败");
        match cmd_check(&input) {
            ChatMessage::Chat => {
                let mut r = chat.clone().start_generate(&input.trim());
                chat.chat_output(&mut r).await;
            }
            ChatMessage::Rollback => {
                let mut r = chat.clone().chat_rollback();
                chat.chat_output(&mut r).await;
            }
            ChatMessage::Switch(id) => {
                println!("{}", id);
                unsafe {
                    match CACHE_MANGER.get().unwrap().get(&id) {
                        // 能够查询到缓存
                        Some(ch) => {
                            chat = Arc::new(Chat::new_chat(id, ch.clone()));
                        }
                        None => {
                            let tmp_cache = Cache::new_cmanger();
                            CACHE_MANGER
                                .get_mut()
                                .unwrap()
                                .insert(id.clone(), tmp_cache.clone());
                            chat = Arc::new(Chat::new_chat(id, tmp_cache));
                        }
                    }
                };
            }
            ChatMessage::Exit => {
                process::exit(0);
            }
            ChatMessage::Error(err) => {
                println!("{}", err);
                process::exit(0);
            }
        }
    }
}
fn cmd_check(info: &str) -> ChatMessage {
    if info.len() == 0 {
        return ChatMessage::Error("输入文本为空，无法推理".to_owned());
    }
    // 判断是否是命令
    if info.chars().next().unwrap() != '>' {
        return ChatMessage::Chat;
    }
    // 用于判断是否是带有数据的命令
    if let Some(index) = info.find(' ') {
        match &info[1..index] {
            "switch" => {
                // 去除结尾的无用字符
                return ChatMessage::Switch(
                    info[index + 1..]
                        .trim_end_matches(['\n', '\r', '\t'])
                        .to_string(),
                );
            }
            _ => {
                return ChatMessage::Error("未知错误，无法推理".to_owned());
            }
        }
    } else {
        match info[1..].trim() {
            "rollback" => {
                return ChatMessage::Rollback;
            }
            "exit" => {
                return ChatMessage::Exit;
            }
            _ => {
                return ChatMessage::Error("未知错误，无法推理".to_owned());
            }
        }
    }
}
