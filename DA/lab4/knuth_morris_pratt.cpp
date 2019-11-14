#include <iostream>
#include <cstring>
#include <vector>

using namespace std;

void prefix_function(const char *image, int *arr, int n){
    arr[0] = 0;
    for (int i = 1; i < n; ++i){
        int current = i - 1;
        while (image[i] != image[arr[current]] && current > 0)
            current = arr[current] - 1;
        if (image[arr[current]] == image[i])
            arr[i] = arr[current] + 1;
        else
            arr[i] = 0;
    }
}

// void prefix_function(const char* image, int *arr, int n){
//     arr[0] = 0;
//     for(int i = 1; i < n; ++i){
//         int current = arr[i - 1];
//         while(image[i] != image[current] && current > 0)
//             current = arr[current - 1];
//         if(image[i] == image[current])
//            arr[i] = current + 1;
//     }
// }

vector<int> KMP_search_all(string needle, string hay){
    int needle_length = needle.size();
    needle += "#" + hay;

    cout << needle << endl;

    int *prefix_table = new int[needle.size()];
    int length = needle.size();
    prefix_function(needle.c_str(), prefix_table, length);

    for(int i = 0; i < length; ++i)
        cout << prefix_table[i];
    cout << endl;

    vector<int> v;
    for(int i = 0; i < length; ++i)
        if(prefix_table[i] == needle_length)
            v.push_back(i - 2 * needle_length);
    delete[] prefix_table;
    return v;
}

int KMP_search(char *needle, char *hay){ /* ВЕРНУТЬСЯ И РАЗОБРАТЬСЯ ПОЧЕМУ МОЖНО НА СТОЛЬКО СДВИГАТЬ (РАЗБИРАЛСЯ!!! ЕСЛИ НЕ ПОМНИШЬ, ТО УЖЕ ЗАБЫЛ, ЛОХ) */
    int needle_length = strlen(needle);
    int hay_length = strlen(hay);
    
    int *prefix_table = new int[needle_length];
    prefix_function(needle, prefix_table, needle_length);

    int needle_pos = 0;
    int hay_pos = 0;

    while(hay_pos <= hay_length - needle_length){
        while(needle[needle_pos] == hay[hay_pos] && needle_pos < needle_length){
            ++needle_pos;
            ++hay_pos;
        }
        if (needle_pos == needle_length){
            delete[] prefix_table; 
            return hay_pos - needle_length;
        }
        if (needle[needle_pos] != hay[hay_pos]){
            if(needle_pos == 0)
                ++hay_pos;
            else
                needle_pos = prefix_table[needle_pos - 1];
        }
    }
    delete[] prefix_table;
    return -1;
}

int main()
{
    char str[] = "abaababaaababaaabaababaaababaaaaabaaaababaa";
    char substr[] = "babaa";

    // cout << KMP_search(substr, str) << endl;

    vector<int> matches = KMP_search_all(substr, str);

    for(int i = 0; i < matches.size(); ++i)
        cout << matches[i] << ' ';
    cout << endl;

    return 0;
}
