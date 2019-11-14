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

int state = RUN;

void sigtstp_handler(int sig);
void sigint_handler(int sig);

int main(int argc, char **argv){

    tree *accounts = NULL;
    if(argc < 2){
        printf("usage: server <port> [account_data]\n");
        exit(EXIT_SUCCESS);
    }
    // подгрузка подобия "базы данных"
    if(argc == 3){
        printf("loading data from %s ...\n", argv[2]);
        FILE *inp = fopen(argv[2], "r"); 
        assert(inp != NULL);
        accounts = deserialize_tree(inp);
        fclose(inp);
    }
    // вешаем обработчики для останова (1) и прерывания (2)
    signal(SIGTSTP, sigtstp_handler);
    signal(SIGINT, sigint_handler);

    print_tree(accounts, 0);

    // формируем адрес
    char adress[30];
    sprintf(adress, "tcp://*:%s", argv[1]);
    printf("adress: %s\n", adress);
    // создаем контекст и подвязываем сокет
    void *context = zmq_ctx_new();
    void *socket = zmq_socket(context, ZMQ_REP);
    int status = zmq_bind(socket, adress);
    assert(status == 0);

    // пока не попросили завершиться
    while (state != CLOSE){

        // если объявлен перерыв
        while (state == PAUSE){
            printf("\nlunch break!");
            sleep(1);
        }

        request req;
        zmq_recv(socket, &req, sizeof(request), 0);

        // если состояние вдруг изменилось на паузу или завершение
        if(state == PAUSE || state == CLOSE)
            continue;

        // ищем клиента в "базе"
        tree *found = find_tree(accounts, req.client1);
        if (found == NULL){
            // если нету -- добавляем
            printf("new client: %s\n", req.client1);
            account temp;
            strcpy(temp.owner, req.client1);
            temp.balance = 0;
            accounts = insert_tree(accounts, &temp);
            // находим добавленное
            found = find_tree(accounts, req.client1);
        }

        response res;

        // совершаем действия и формируем ответ
        switch (req.oper){
        case DEPOSIT:
            printf("%s deposited %d\n", req.client1, req.value);
            found->acc.balance += req.value;
            res.stat = OK;
            res.value = found->acc.balance;
            break;
        case WITHDRAW:
            printf("%s withdrawed %d\n", req.client1, req.value);
            int *balance = &found->acc.balance;
            if (*balance < req.value){
                res.stat = NOT_OK;
            }
            else{
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
            else{
                int *balance = &found->acc.balance;
                if (*balance < req.value)
                    res.stat = NOT_OK;
                else{
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

        // если в состаянии обработки, то имитируем сложные оберации в банке
        if(state == RUN)
            // sleep(10);
        
        // отправляем ответ
        status = zmq_send(socket, &res, sizeof(response), 0);
    }
    // если данные брали из базы данных, то обновляем их там
    if (argc == 3){
        printf("\nsaving data to %s ...\n", argv[2]);
        FILE *op = fopen(argv[2], "w");
        assert(op != NULL);
        serialize_tree(accounts, op);
        fclose(op);
    }

    // убираем за собой
    destroy_tree(accounts);
    zmq_close(socket);
    zmq_ctx_destroy(context);

    return 0;
}

void sigtstp_handler(int sig){
    // если работаем, то объявляем перерыв, иначе начинаем работать
    if (state == RUN)
        state = PAUSE;
    else
        state = RUN;
}

void sigint_handler(int sig){
    // пора закругляться
    state = CLOSE;
}