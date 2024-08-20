use std::{io::{self, BufRead, Write}, sync::Arc};


use super::chat::Chat;


pub  fn cmd_server(c:Chat){
    let stdin = io::stdin();
    let mut handle = stdin.lock();
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
                let v=c.chat_rollback();
                print!("{:?}",c.decode(&v));
            }
            _ => {
                let v=c.start_generate(input.trim());
                print!("{:?}",v);
            }
        }
    }
 
}