#ifndef REQUEST_H
#define REQUEST_H

enum{
    DEPOSIT,
    WITHDRAW,
    TRANSFER,
    BALANCE
};

typedef struct{
    char client1[20];
    char client2[20];
    int oper;
    int value;
} request;

#endif