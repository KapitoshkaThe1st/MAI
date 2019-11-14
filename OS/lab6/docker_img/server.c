// Вариант 30   3 2 2
// Бинарное дерево, где ключом является идентификатор клиента
// Строка
// Возможность временной приостановки работы сервера без выключения.
// Сообщения серверу можно отправлять, но ответы сервер не отправляет до возобновления работы

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <zmq.h>
#include <assert.h>
#include <signal.h>

#include "request.h"
#include "response.h"
#include "binary_tree.h"

#define RUN 0
#define PAUSE 1
#define CLOSE 2

int control = RUN;

void sigtstp_handler(int sig);
void sigint_handler(int sig);

int main(int argc, char **argv){
    tree *accounts = NULL;
    if(argc < 2){
        printf("usage: server <port> [account_data]\n");
        exit(EXIT_SUCCESS);
    }
    if(argc == 3){
        printf("loading data from %s ...\n", argv[2]);
        FILE *inp = fopen(argv[2], "r"); 
        assert(inp != NULL);
        accounts = deserialize_tree(inp);
        fclose(inp);
    }

    signal(SIGTSTP, sigtstp_handler);
    signal(SIGINT, sigint_handler);

    print_tree(accounts, 0);

    char adress[30];
    sprintf(adress, "tcp://*:%s", argv[1]);
    // printf("adress: %s\n", adress);

    void *context = zmq_ctx_new();
    void *socket = zmq_socket(context, ZMQ_REP);
    int status = zmq_bind(socket, adress);
    assert(status == 0);

    int req_count = 0;

    while (1){
        request req;
        zmq_recv(socket, &req, sizeof(request), 0);
        // zmq_recv(socket, &req, sizeof(request), ZMQ_DONTWAIT);

        // printf("client1: %s\n", req.client1);
        // printf("client2: %s\n", req.client2);
        // printf("oper: %d\n", req.oper);
        // printf("value: %d\n", req.value);

        tree *found = find_tree(accounts, req.client1);
        if (found == NULL)
        {
            printf("new client: %s\n", req.client1);
            account temp;
            strcpy(temp.owner, req.client1);
            temp.balance = 0;
            accounts = insert_tree(accounts, &temp);
        }
        found = find_tree(accounts, req.client1);

        response res;

        switch (req.oper)
        {
        case DEPOSIT:
            printf("%s deposited: %d\n", req.client1, req.value);
            found->acc.balance += req.value;
            res.stat = OK;
            res.value = found->acc.balance;
            break;
        case WITHDRAW:
            printf("%s withdrawed %d\n", req.client1, req.value);
            int *balance = &found->acc.balance;
            if (*balance < req.value)
            {
                res.stat = NOT_OK;
            }
            else
            {
                *balance -= req.value;
                res.stat = OK;
            }
            res.value = *balance;
            break;
        case TRANSFER:
            printf("%s transfered %d to %s\n", req.client1, req.value, req.client2);
            tree *targ = find_tree(accounts, req.client2);
            if (targ == NULL)
                res.stat = NOT_OK;
            else
            {
                int *balance = &found->acc.balance;
                if (*balance < req.value)
                    res.stat = NOT_OK;
                else
                {
                    *balance -= req.value;
                    targ->acc.balance += req.value;
                    res.stat = OK;
                }
                res.value = *balance;
            }
            break;
        case BALANCE:
            printf("%s requested balance\n", req.client1);
            res.stat = OK;
            res.value = found->acc.balance;
            break;
        default:
            res.stat = NOT_OK;
            res.value = found->acc.balance;
            break;
        }

        if(control == RUN)
            sleep(10);
        zmq_send(socket, &res, sizeof(response), 0);

        while (control == PAUSE){
            printf("lunch break!\n");
            sleep(1);
        }

        if (control == CLOSE)
            break;
    }
    if (argc == 3){
        printf("saving data to %s ...\n", argv[2]);
        FILE *op = fopen(argv[2], "w");
        assert(op != NULL);
        serialize_tree(accounts, op);
        fclose(op);
    }

    return 0;
}

void sigtstp_handler(int sig){
    if (control == RUN)
        control = PAUSE;
    else
        control = RUN;
}

void sigint_handler(int sig){
    control = CLOSE;
}