#ifndef DATABASE_H
#define DATABASE_H

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#define UNAME_LEN_LIM 20

typedef struct{
    char uname[UNAME_LEN_LIM];
    int wins;
    int losses;
    int lefts;
} record;

typedef struct{
    record *data;
    int count;
}database;

database init(){
    database res = {NULL, 0};
    return res;
}

database load(char *filename){
    struct stat file_stat;

    int fd = open(filename, O_RDONLY);
    assert(fd > 0);

    int status = fstat(fd, &file_stat);
    assert(status == 0);

    database res;
    res.data = (record*)malloc(file_stat.st_size);
    read(fd, res.data, file_stat.st_size);
    
    res.count = file_stat.st_size / sizeof(record);

    close(fd);
    return res;
}

void save(char *filename, database *base){
    int fd = open(filename, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
    assert(fd > 0);

    write(fd, base->data, base->count * sizeof(record));

    close(fd);
}

void add(database *base, record rec){
    if(base->data != NULL)
        base->data = (record*)realloc(base->data, (base->count + 1) * sizeof(record));
    else
        base->data = (record*)malloc(sizeof(record));
    base->data[base->count] = rec;
    base->count++;
}

record* get_rec(database *base, int index){
    return &base->data[index];
}

record* find_rec(database *base, char *name){
    for(int i = 0; i < base->count; i++)
        if(!strcmp(name, base->data[i].uname))
            return &base->data[i];
    return NULL;
}

#endif