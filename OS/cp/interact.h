#ifndef INTERACT_H
#define INTERACT_H

#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>

#define MT_EMPTY 0
#define MT_LOGIN 1
#define MT_SHOT 2
#define MT_FIELD 3
#define MT_TURN 4
#define MT_HIT 5
#define MT_MISS 6
#define MT_WIN 7
#define MT_LOSE 8
#define MT_RATING 9
#define MT_ERROR 10

#define SR_MISS 0
#define SR_HIT 1

typedef struct{
    int type;
    int id;
    int size;
    void *buffer;
}msg_t;


typedef struct{
    int x;
    int y;
}shot;

void init_msg(msg_t *m){
    m->type = MT_EMPTY;
    m->size = 0;
    m->buffer = NULL;
}

void set_type_msg(msg_t *m, int type){
    m->type = type;
}

void set_id_msg(msg_t *m, int id){
    m->id = id;
}

void init_size_msg(msg_t *m, int size){
    m->buffer = malloc(size);
    m->size = size;
}

void close_msg(msg_t  *m){
    if(m->buffer != NULL)
        free(m->buffer);
    m->size = 0;
}

void init_data_msg(msg_t *m, void *data, int size, int type, int id){
    init_size_msg(m, size);
    memcpy(m->buffer, data, size);
    set_type_msg(m, type);
    set_id_msg(m, id);
}

int send_msg(int socket, msg_t *m){
    send(socket, &m->size, sizeof(int), 0);
    send(socket, &m->type, sizeof(int), 0);
    send(socket, &m->id, sizeof(int), 0);
    send(socket, m->buffer, m->size, 0);

    return 1;
}

int recieve_msg(int socket, msg_t *m){
    if(recv(socket, &m->size, sizeof(int), 0) == 0)
        return 0;
    init_size_msg(m, m->size);
    recv(socket, &m->type, sizeof(int), 0);
    recv(socket, &m->id, sizeof(int), 0);
    recv(socket, m->buffer, m->size, 0);

    return 1;
}

#endif