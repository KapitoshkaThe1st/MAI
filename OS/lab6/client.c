#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <zmq.h>

#include "request.h"
#include "response.h"

void show_menu();

int main(int argc, char **argv){
    if(argc != 3){
        printf("usage: client <port> <client_name>\n");
        exit(EXIT_SUCCESS);
    }

    // формируем адрес
    char adress[30];
    sprintf(adress, "tcp://localhost:%s", argv[1]);
    printf("adress: %s\n", adress);

    // создаем контекст и коннектимся
    void *context = zmq_ctx_new();
    void *socket = zmq_socket(context, ZMQ_REQ);
    int status = zmq_connect(socket, adress);
    assert(status == 0);

    char buffer[20];

    printf("ask <help> if you don't know what to do\n");

    // пока не попросили завершиться
    while(1){
        scanf("%s", buffer);
        if(!strcmp("exit", buffer))
            break;
        if (!strcmp("help", buffer)){
            show_menu();
            continue;
        }

        // для проверки на доступность операции
        int correct = 0;
        request req;

        // выбираем операцию
        strcpy(req.client1, argv[2]);
        if(!strcmp("deposit", buffer)){
            req.oper = DEPOSIT;
            scanf("%d", &req.value);
            correct = 1;
        }
        if(!strcmp("withdraw", buffer)){
            req.oper = WITHDRAW;
            scanf("%d", &req.value);
            correct = 1;
        }
        if(!strcmp("transfer", buffer)){
            req.oper = TRANSFER;
            scanf("%s", req.client2);
            scanf("%d", &req.value);
            correct = 1;
        }
        if(!strcmp("balance", buffer)){
            req.oper = BALANCE;
            req.value = 0;
            correct = 1;
        }
        if(correct){
            // если корректная операция
            printf("Operation accepted. Waiting for completion.\n");
            // отправляем запрос
            zmq_send(socket, &req, sizeof(request), 0);

            response res;
            // ждем ответ
            zmq_recv(socket, &res, sizeof(response), 0);

            // проверка статуса ответа
            switch (res.stat){
                case OK:
                    printf("Status: OK. Operation completed.\n");
                    break;
                case NOT_OK:
                    printf("Status: NOT_OK. Operation denied.\n");
                    break;
            }
            printf("Now your balance is: %d\n", res.value);
        }
        else{
            printf("Wrong request. Try again please\n");
        }
    }

    // убираемся за собой
    zmq_close(socket);
    zmq_ctx_destroy(context);

    return 0;
}

void show_menu(){
    printf("You can:\n");
    printf("deposit <sum>\n");
    printf("withdraw <sum>\n");
    printf("transfer <person> <sum>\n");
    printf("balance\n");
    printf("exit\n");
}