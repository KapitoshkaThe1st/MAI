#!/bin/bash

if [ $# -lt 2 ]
then
    echo "usage run <port> [acc_data]"
    exit 0
fi

port=$1
path=accounts.bnk
abs_path=$(realpath $2)

# echo "abs_path: " $abs_path 
# echo "port: " $port
# --rm --name server

echo "Server is running now..."
sudo docker run -p $port:80 -v $abs_path:/$path proba/proba