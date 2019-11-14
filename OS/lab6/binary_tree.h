#ifndef BINARY_TREE_H
#define BINARY_TREE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char owner[20];
    int balance;
} account;

typedef struct node_t{
    struct node_t *left, *right;
    account acc;
} tree;

tree* create_tree(account *acc){
    tree *temp = (tree*)malloc(sizeof(tree));
    temp->left = temp->right = NULL;
    temp->acc = *acc;
    return temp;
}

tree *insert_tree(tree *t, account *acc){
    if(t == NULL)
        return create_tree(acc);
    else if(strcmp(acc->owner, t->acc.owner) < 0)
        t->left = insert_tree(t->left, acc);
    else if(strcmp(acc->owner, t->acc.owner) > 0)
        t->right = insert_tree(t->right, acc);
    return t;
}

tree *find_tree(tree *t, char *name){
    while(t != NULL && strcmp(name, t->acc.owner) != 0){
        if(strcmp(name, t->acc.owner) < 0)
            t = t->left;
        else
            t = t->right;   
    }
    return t;
}

tree* detach_min(tree* t){
    tree *par = t;
    tree *temp = t->right;
    while(temp->left != NULL){
        par = temp;
        temp = temp->left;
    }
    if(par == t)
        par->right = temp->right;
    else
        par->left = temp->right;
    return temp;
}

tree *erase_tree(tree *t, char *name){
    if(t == NULL)
        return NULL;
    else if (strcmp(name, t->acc.owner) < 0)
        t->left = erase_tree(t->left, name);
    else if (strcmp(name, t->acc.owner) > 0)
        t->right = erase_tree(t->right, name);
    else{
        tree *temp;
        if (t->left == NULL && t->right)
            temp = t->right;
        else if(t->right == NULL)
            temp = t->left;
        else{
            tree *m = detach_min(t);
            t->acc = m->acc;
            free(m);
            return t;
        }
        free(t);
        return temp;
    }
    return t;
}

void print_tree(tree *t, int tab){
    if(t == NULL)
        return;
    print_tree(t->left, tab + 2);
    for(int i = 0; i < tab; i++)
        printf("%c", ' ');
    printf("%s : %d \n", t->acc.owner, t->acc.balance);
    print_tree(t->right, tab + 2);
}

void destroy_tree(tree *t){
    if(t->left != NULL)
        destroy_tree(t->left);
    if (t->right != NULL)
        destroy_tree(t->right);
    free(t);
    t = NULL;
}

void serialize_tree(tree *t, FILE *file){
    if(t == NULL)
        return;
    fwrite((void*)&t->acc, sizeof(account), 1, file);
    serialize_tree(t->left, file);
    serialize_tree(t->right, file);
}

tree* deserialize_tree(FILE *file){
    tree *temp = NULL;
    account acc;
    while(fread(&acc, sizeof(account), 1, file))
        temp = insert_tree(temp, &acc);
    return temp;
}

#endif