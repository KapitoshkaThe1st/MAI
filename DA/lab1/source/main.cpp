#include <iostream>
#include "vector.hpp"
#include "queue.hpp"

class TRecord
{
public:
    char *key;
    char *value;

    TRecord() :  key(nullptr), value(nullptr) {

    }

    void Print() {
        std::cout << key[0] << ' ' << key[1] << key[2] << key[3] << ' ' << key[4] << key[5] << '\t' << value << std::endl;
    }
};

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

void RadixSort(TVector<TRecord> &recs, int count) {
    const int digsCount = 6; // количество разрядов
    const int letterRange = 26; // 26 букв латинского алфавита
    const int numberRange = 10; // 10 циыфр

    TRecord **recPtrs = new TRecord*[count];

    for (int i = 0; i < count; ++i) {
        recPtrs[i] = &recs[i];
    }

    TQueue<TRecord*> letterQueue[letterRange];
    TQueue<TRecord*> numberQueue[numberRange];

    const int numsBeg = 1;
    const int numsEnd = 3;

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
}

void ReadRecord(TRecord &rec) {
    const int bufferSize = 2049;
    const int keySize = 7;

    rec.key = new char[keySize];

    // Формат ввода: [A 123 BC]<tab>[значение(строка до 2048 значащих символов)]    
    std::cin >> rec.key[0];
    std::cin.get();
    std::cin >> rec.key[1];
    std::cin >> rec.key[2];
    std::cin >> rec.key[3];
    std::cin.get();
    std::cin >> rec.key[4];
    std::cin >> rec.key[5];
    std::cin.get();
    rec.key[6] = '\0';

    char buffer[bufferSize];

    std::cin.getline(buffer, bufferSize);
    int count = std::cin.gcount();

    if(std::cin.fail()) { // если cin.getline(buff, N) считывает N-1 символов, не наткнувшись на разделитель (по-умолчанию '\n'), то выбрасывается cin.fail,
                        // но если N-й символ это разделитель (по-умолчанию '\n'), то cin.fail не выбрасывается
        std::cin.clear();
        std::cin.ignore(bufferSize, '\n');
        ++count; // т.к. cin.getline прочитал свои N символов, и среди них не оказалось разделителя, то он его и не посчитает, поэтому добавляем 1 в этом случае
    }

    rec.value = new char[count];
    for (int i = 0; i < count; ++i) {
        rec.value[i] = buffer[i];
    }
}

TRecord ReadRecord() {
    TRecord rec;
    ReadRecord(rec);
    return rec;
}

int main()
{
    if(std::cin.peek() != EOF) {

        TVector<TRecord> recs(10);

        while (!std::cin.eof()) {
            while(std::cin.peek() == '\n'){
                std::cin.get();
            }
            TRecord temp = ReadRecord();
            recs.PushBack(temp);
            while(std::cin.peek() == '\n'){
                std::cin.get();
            }
            if(std::cin.peek() == EOF)
                break;
        }

        RadixSort(recs, recs.Size());

        for (int i = 0; i < recs.Size(); ++i) {
            recs[i].Print();
            delete [] recs[i].key;
            delete [] recs[i].value;
        }
    }

    return 0;
}