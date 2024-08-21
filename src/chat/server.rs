use std::{io::{self, BufRead, Write}, sync::{Arc, Mutex}};

use crate::{cache::{Cache, CACHE_MANGER}, chat::chat::Chat, MY_LLAMA};

pub  fn cmd_server(){
    let stdin = io::stdin();
    let mut handle = stdin.lock();
    // 限制作用域
    {
        let c=unsafe { CACHE_MANGER.get_mut().unwrap() };
        c.insert("1".to_owned(), Arc::new(Mutex::new(Cache::new(MY_LLAMA.get().unwrap().new_cache()))));
        
    }

    //  创建一个可安全共享的引用
    loop {
        print!("请输入推理文本 (输入 'exit' 退出程序): ");
        io::stdout().flush().unwrap(); // 确保输出立即刷新

        let mut input = String::new();
        handle.read_line(&mut input).expect("读取失败");

        match input.trim() {
            "exit" => {
                println!("退出程序...");
                break;
            }
            ">rollback"=>{
                // let v=c.chat_rollback();
                // print!("{:?}",c.decode(&v));
            }
            _ => {
                let c=Chat::new_chat("1".to_owned(), unsafe { CACHE_MANGER.get().unwrap().get("1").unwrap().clone() });
                let v=c.start_generate(input.trim());
                print!("{:?}",v);
            }
        }
    }
 
}