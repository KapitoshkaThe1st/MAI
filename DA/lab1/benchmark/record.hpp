#ifndef RECORD_H
#define RECORD_H

#include <iostream>

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

void ReadRecord(TRecord &rec) {
    const int bufferSize = 2049; // 2048 значащих, 1 -- под терминатор
    const int keySize = 7; // 6 значащих, 1 -- под термиантор

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

#endif