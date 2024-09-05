use crate::{
    //todo f16代码
    //     cache::{self, Cache, CACHE_MANGER_F16 as CACHE_MANGER}
    cache::{self, Cache, CACHE_MANGER_F32 as CACHE_MANGER},
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
    Rollback(usize),
    Switch(String),
    Exit,
    Error(String),
    Help,
}
pub async fn cmd_server() {
    let stdin = io::stdin();
    let mut handle = stdin.lock();
    // f16代码
    // let mut chat = Arc::new(Chat::<f16>::new_chat("1".to_owned(), cache::Cache::<f16>::new_cmanger()));
    // 初始化默认Chat
    let mut chat = Arc::new(Chat::<f32>::new_chat("1".to_owned(), cache::Cache::<f32>::new_cmanger()));
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
            ChatMessage::Rollback(i)=> {
                let r = chat.clone().chat_rollback(i);
                match r {
                    Ok(_) =>{},
                    Err(e) => {
                        println!("{}", e);
                        continue;
                    }
                }
                chat.chat_output(&mut r.unwrap()).await
            }
            ChatMessage::Switch(id) => {
                unsafe {
                    match CACHE_MANGER.get().unwrap().get(&id) {
                        // 能够查询到缓存
                        Some(ch) => {
                            println!("查询到缓存，转换到chat {}", id);
                            // f16代码
                            //  chat = Arc::new(Chat::<f16>::new_chat(id, ch.clone()));

                            chat = Arc::new(Chat::<f32>::new_chat(id, ch.clone()));
                          // 输出历史内容
                            chat.decode(&ch.lock().unwrap().get_info());
                        }
                        None => {
                            println!("未查询到缓存，新生成chat {}", id);
                            // f16代码
                            // let tmp_cache = Cache::<f16>::new_cmanger();
                            let tmp_cache = Cache::<f32>::new_cmanger();
                            CACHE_MANGER
                                .get_mut()
                                .unwrap()
                                .insert(id.clone(), tmp_cache.clone());
                            // f16代码
                            // chat = Arc::new(Chat::<f16>::new_chat(id, tmp_cache));
                            chat = Arc::new(Chat::<f32>::new_chat(id, tmp_cache));
                        }
                    }
                };
            }
            ChatMessage::Exit => {
                println!("程序正常退出，欢迎下次再见");
                process::exit(0);
            }
            ChatMessage::Help => {
                println!(
                    ">list           列出现存的会话及对话次数
                    >rollback <nums>        会话回滚
                    >switch <id>    切换至指定会话
                    >help           打印帮助信息
                    使用 /exit 或 Ctrl + C 结束程序"
                )
            }
            ChatMessage::Error(err) => {
                println!("{} 程序退出", err);
                process::exit(0);
            }
        }
    }
}
fn cmd_check(info: &str) -> ChatMessage {
    if info.len() == 0 {
        return ChatMessage::Error("请在操作符后面添加数字".to_owned());
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
            "rollback" => {
                
                let num = info[index + 1..]
                 .trim_end_matches(['\n', '\r', '\t'])
                 .to_string().parse::<usize>();
                match num {
                    Ok(data) =>   return ChatMessage::Rollback(data),
                    Err(e) => return ChatMessage::Error(format!("{}",e)),
                }
              
            }
            _ => {
                return ChatMessage::Error("位置命令，不支持".to_owned());
            }
        }
    } else {
        match info[1..].trim() {
            "exit" => {
                return ChatMessage::Exit;
            }
            "help"=>return  ChatMessage::Help,
            _ => {
                return ChatMessage::Error("未知错误，无法推理".to_owned());
            }
        }
    }
}