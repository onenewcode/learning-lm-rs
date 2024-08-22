use std::{
    io::{self, BufRead, Write},
    process,
    sync::{Arc, Mutex},
};

use crate::{
    cache::{Cache, CACHE_MANGER},
    chat::chat::Chat,
    MY_LLAMA,
};
enum ChatMessage {
    Chat,
    Rollback,
    Switch(String),
    Exit,
    Error(String),
}
pub fn cmd_server() {
    let stdin = io::stdin();
    let mut handle = stdin.lock();
    // 限制作用域
    {
        let c = unsafe { CACHE_MANGER.get_mut().unwrap() };
        c.insert(
            "1".to_owned(),
            Arc::new(Mutex::new(Cache::new(MY_LLAMA.get().unwrap().new_cache()))),
        );
    }
    // 初始化默认Chat
    let c = Chat::new_chat("1".to_owned(), unsafe {
        CACHE_MANGER.get().unwrap().get("1").unwrap().clone()
    });
    //  创建一个可安全共享的引用
    loop {
        print!("请输入推理文本 (输入 '>exit' 退出程序): ");
        io::stdout().flush().unwrap(); // 确保输出立即刷新

        let mut input = String::new();
        handle.read_line(&mut input).expect("读取失败");
        match cmd_check(&input) {
            ChatMessage::Chat => {
                let result = c.start_generate(&input.trim());
                println!("{}", result);
            }
            ChatMessage::Rollback => {
                let result = c.chat_rollback();
                println!("{}", c.decode(&result));
            }
            ChatMessage::Switch(id) => {
                println!("{}", id);
                // if unsafe { CACHE_MANGER.get().unwrap().get(&id).is_some}
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
                return ChatMessage::Switch(info[index + 1..].to_owned());
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
