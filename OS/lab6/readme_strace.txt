strace -e trace=socket,bind,listen,send,recv -o trace ./server 8888 accounts.bnk 
