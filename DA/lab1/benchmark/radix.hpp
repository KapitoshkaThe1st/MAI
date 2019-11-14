#ifndef RADIX_H
#define RADIX_H

#include "vector.hpp"
#include "queue.hpp"
#include "record.hpp"

class Record;

int IsDigit(char ch) {
    return ch >= '0' && ch <= '9';
}
int IsLetter(char ch) {
    return ch >= 'A' && ch <= 'Z';
}

int SymbIndex(char ch) {
    if (IsDigit(ch)) {
        return ch - '0';
    }
    else if (IsLetter(ch)) {
        return ch - 'A';
    }
    else {
        return -1;
    }
}

/*void RadixSort(TVector<TRecord> &recs, int count) {
    const int digsCount = 6; // количество разрядов
    const int letterRange = 26; // 26 букв латинского алфавита
    const int numberRange = 10; // 10 циыфр

    TRecord **recPtrs = new TRecord*[count];

    for (int i = 0; i < count; ++i) {
        recPtrs[i] = &recs[i];
    }

    TQueue<TRecord*> letterQueue[letterRange];
    TQueue<TRecord*> numberQueue[numberRange];

    const int numsBeg = 1; // начальная позиция цифр в ключе
    const int numsEnd = 3; // конечная позиция цифр в ключе

    for (int curDig = digsCount - 1; curDig >= 0; --curDig) { // для каждого разряда
        if(curDig >= numsBeg && curDig <= numsEnd) { 
            for (int i = 0; i < count; ++i) {
                numberQueue[SymbIndex(recPtrs[i]->key[curDig])].Push(recPtrs[i]);
                // разбиваем по карманам
            }

            int index = 0;

            for (int k = 0; k < numberRange; ++k) { // собираем из карманов
                while (!numberQueue[k].Empty()) {
                    recPtrs[index] = numberQueue[k].Pop();
                    ++index;                
                }
            }
        }
        else {
            for (int i = 0; i < count; ++i) {
                letterQueue[SymbIndex(recPtrs[i]->key[curDig])].Push(recPtrs[i]);
                // разбиваем по карманам
            }

            int index = 0;

            for (int k = 0; k < letterRange; ++k) { // собираем из карманов
                while (!letterQueue[k].Empty()) {
                    recPtrs[index] = letterQueue[k].Pop();
                    ++index;                
                }
            }
        }
    }

    TRecord *newRecs = new TRecord[recs.Size()];

    for (int i = 0; i < count; ++i) {
        newRecs[i] = *recPtrs[i];
    }
    
    delete[] recPtrs;
    delete[] recs.data;

    recs.data = newRecs;
}*/

void RadixSort(TRecord *recs, int count) {
    const int digsCount = 6; // количество разрядов
    const int letterRange = 26; // 26 букв латинского алфавита
    const int numberRange = 10; // 10 цифр

    TQueue<TRecord> letterQueue[letterRange];
    TQueue<TRecord> numberQueue[numberRange];

    const int numsBeg = 1; // начальная позиция цифр в ключе
    const int numsEnd = 3; // конечная позиция цифр в ключе

    for (int curDig = digsCount - 1; curDig >= 0; --curDig) { // для каждого разряда
        if(curDig >= numsBeg && curDig <= numsEnd) { 
            for (int i = 0; i < count; ++i) {
                numberQueue[SymbIndex(recs[i].key[curDig])].Push(recs[i]);
                // разбиваем по карманам
            }

            int index = 0;

            for (int k = 0; k < numberRange; ++k) { // собираем из карманов
                while (!numberQueue[k].Empty()) {
                    recs[index] = numberQueue[k].Pop();
                    ++index;                
                }
            }
        }
        else {
            for (int i = 0; i < count; ++i) {
                letterQueue[SymbIndex(recs[i].key[curDig])].Push(recs[i]);
                // разбиваем по карманам
            }

            int index = 0;

            for (int k = 0; k < letterRange; ++k) { // собираем из карманов
                while (!letterQueue[k].Empty()) {
                    recs[index] = letterQueue[k].Pop();
                    ++index;                
                }
            }
        }
    }
}

#endif