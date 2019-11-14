#ifndef RESPONSE_H
#define RESPONSE_H

enum{
    OK,
    NOT_OK
};

typedef struct{
    int stat;
    int value;
} response;

#endif