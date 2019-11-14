#include <iostream>
#include <algorithm>
#include <cstring>

using std::cout;
using std::cin;
using std::endl;

/*void z_block(const char *str, char *z, int n){
    int l = 0;
    for(int i = 0; i < n; ++i)
        z[i] = 0;

    for(int i = 1; i < n; ++i){
        if(i < l + z[l]){ // если внутри какого-то уже найденного z-блока
            int k = i - l;
            if(z[l] > k + z[k]){    // и если z-блок образа k не выходит из z-блока в котором находимся
                z[i] = z[k];    // то просто длину z-блока для i принимаем за длину z-блока k
            }
            else{   // иначе начинаем сравнивать уже вне z-блока
                int c = 0;
                while(str[l + c] == str[l+z[l]+c])
                    ++c;
                z[i] = z[k] + c;
            }
        }
        else{   // если не внутри z-блока, то просто ищем сколько совпало и помечаем до кудова добрались
            int q = 0;
            while(str[q+i] == str[q] && q+i < n)
                ++q;
            z[i] = q;
            l = i;
        }
    }
}*/

/*void z_block(const char *str, char *z, int n)
{
    int l = 0;
    int r = 0;
    for (int i = 0; i < n; ++i)
        z[i] = 0;

    for (int i = 1; i < n; ++i)
    {
        cout << "i = " << i << endl;
        cout << "l = " << l << " r = " << r << endl;
        if (i <= r)
        { // если внутри какого-то уже найденного z-блока
            int k = i - l;
            if (z[l] >= k + z[k]){   // и если z-блок образа k не выходит из z-блока в котором находимся
                z[i] = z[k]; // то просто длину z-блока для i принимаем за длину z-блока k
            }
            else
            { // иначе начинаем сравнивать уже вне z-блока
                cout << "here!" << endl;
                int c = 0;
                while (str[z[l] + c] == str[l + z[l] + c]){
                    ++c;
                }
                z[i] = z[k] + c;
                l = i;
                r = i + z[i] - 1;
            }
        }
        else
        { // если не внутри z-блока, то просто ищем сколько совпало и помечаем до кудова добрались
            int q = 0;
            while (str[q + i] == str[q] && q + i < n)
                ++q;
            z[i] = q;
            if(q){
                l = i;
                r = i + z[i] - 1;
            }
        }
    }
}*/

int min(int a, int b){
    return a < b ? a : b;
}

// void z_block(const char *str, char *z, int n){
//     for(int i = 0; i < n; ++i)
//         z[i] = 0;
//     int l = 0, r = 0; 

//     for(int i = 1; i < n; ++i){
//         if(i <= r){
//             int k = l - i;
//             int s = min(z[k], r - i + 1); 
//             int c = 0;
//             while(str[s+c] == str[i+s+c])
//                 ++c;
//             z[i] = s + c;
//             r = i + z[i] - 1;
//             l = i;
//         }
//         else{
//             int c = 0;
//             while (str[c] == str[i + c])
//                 ++c;
//             z[i] = c;
//             if(z[i]){
//                 r = i + z[i] - 1;
//                 l = i;
//             }
//         }
//     }
// }

void z_block(const char *str, char *z, int n)
{
    for (int i = 0; i < n; ++i)
        z[i] = 0;
    int l = 0, r = 0;

    for (int i = 1; i < n; ++i)
    {
        int c = 0;	// количество совпавших
        int s = 0;	// стартовая позиция для сравнения
		int k = i - l;	// куда отобразится i-й элемент в своем z-блоке
        if (i <= r)	// если i в каком-то z-блоке
            int s = min(z[k], r - i + 1);	// за старт принимаем минимальный из z[k] и длины остатка z-блока 
											// т.к. z[k] может выходить за границы z-блока, а там уже нужно сравнивать
        while (i+c < n &&str[c + s] == str[i + c + s])	// ищем количество совпавших
           ++c;
        z[i] = s + c;
        if (z[i])	// если совпало больше 0, то нашелся z-блок, поэтому сдвигаем границы
        {
            r = i + z[i] - 1;
            l = i;
        }
    }
}
template<typename T>
void reverse(T *str, size_t n){
    for (int i = 0; i < n / 2; ++i){
        T temp = str[i];
        str[i] = str[n - i - 1];
        str[n - i - 1] = temp;
    }
}

void preprocessing_PHS(char *str, char *n_, char *l, int n){
    for(int i = 0; i < n; ++i)
        l[i] = 0;
    reverse(str, n);    // n-функция для строки ищется как z-функция для инвертированной строки, с последующим инвертированием найденного 
    z_block(str, n_, n);
    reverse(str, n);
    reverse(n_, n);
    for(int i = 0; i < n; ++i){ // l[j] -- самое правое окончание подстроки длины n[i] равной некоторому суффиксу 
                                // всей строки, при этом предшествующий ей символ не равен предшествующему для суффикса
        int j = n - n_[i];
        l[j] = i;
    }
}

void preprocessing_PPS(const char *str, char *r, int n){
    for(int i = 0; i < 256; ++i)
        r[i] = 0;
    for(int i = 0; i < n; ++i)
        r[str[i]] = i;
}

int boyer_moore_search(char *needle, const char *hay, int m, int n){

    /* не работает :( возможно потому что нету для неполного вхождения совпавшей части. Надо искать. */

    char symbols[256];
    char *n_func = new char[m];
    char *l_func = new char[m];
    preprocessing_PPS(needle, symbols, m);
    preprocessing_PHS(needle, n_func, l_func, m);

    for (int i = 0; i < m; ++i)
        cout << needle[i] << ' ';
    cout << endl;
    for(int i = 0; i < m; ++i)
        cout << (int)l_func[i] << ' ';
    cout << endl;
    /* еще в препроцессинге нужно найти функцию p на случай если ПХС и слева нет полного вхождения совпавшей части */

    for (int i = 0; i < m; ++i)
        cout << static_cast<int>(symbols[needle[i]]) << ' ';
    cout << endl;

    for(int i = 0; i < n;){
        cout << "i: " << i << endl;
        int j = m - 1;
        while(hay[i + j] == needle[j])
            j--;
        if(j == 0)
            return i;
        else{
            int shift = min(l_func[i], symbols[hay[i + j]]);
            cout << "shift: " << shift << endl; 
            i += m - shift -1;
        }
    }

    return 0;
}

#define array_length(X) sizeof(X)/sizeof(X[0])-1

int main(int argc, char const *argv[]){
    char str[] = "abcadcabbacdacabdabdabcadbacb";
    char substr[] = "cabdabdab";
    int match = boyer_moore_search(substr, str, array_length(substr), array_length(str));
    cout << "match: " << match << endl; 

    return 0;
}