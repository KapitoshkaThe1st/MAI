#ifndef QUEUE_H
#define QUEUE_H

template <typename T>
struct Node {
    T value;
    Node *next;
};

template <typename T>
class TQueue {
private:
    Node<T> *head;
    Node<T> *tail;
    int size;
public:
    TQueue() : size(0)
    {
        tail = head = new Node<T>;
    }
    ~TQueue() {
        while (!Empty()) {
            Pop();
        }
        delete head;
    }
    void Push(T el) {
        tail->next = new Node<T>;
        tail->value = el;
        tail = tail->next;
        ++size;
    }
    T Fetch() {
        return head->value;
    }
    T Pop() {
        Node<T> *temp = head;
        head = head->next;
        T toReturn = temp->value;
        delete temp;
        --size;
        return toReturn;
    }
    int Empty() {
        return head == tail;
    }
};

#endif