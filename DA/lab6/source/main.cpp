#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <cassert>
#include <iomanip>

typedef long long digit_type;
const int base = 100000000;
const int baseLen = (int)(log(base) / log(10));

class TBigInt{
public:
    TBigInt() : digits(std::vector<digit_type>(1,0)) {}
    TBigInt(const TBigInt &oth) : digits(oth.digits) {}
    TBigInt(TBigInt &&oth){
        digits = std::move(oth.digits);
    }
    TBigInt(const size_t size, size_t) : digits(std::vector<digit_type>(size)) {}
    TBigInt(std::string str){
        size_t len = str.length();
        for(size_t i = len - baseLen; i < len; i -= baseLen){
            assert(i < len);
            digits.push_back(atoll(str.c_str() + i));
            str[i] = 0;
        }
        if(len % baseLen != 0){
            digits.push_back(atoll(str.c_str()));
        }
        FilterZeros();
    }
    TBigInt(std::istream &istr){
        std::string inputBuffer;
        istr >> inputBuffer;

        size_t len = inputBuffer.length();
        for(size_t i = len - baseLen; i < len; i -= baseLen){
            assert(i < len);
            digits.push_back(atoll(inputBuffer.c_str() + i));
            inputBuffer.resize(i);
        }
        if(len % baseLen != 0){
            digits.push_back(atoll(inputBuffer.c_str()));
        }
        FilterZeros();
    }

    TBigInt(digit_type num) : digits(std::vector<digit_type>()){
        if(num == 0){
            digits.push_back(0);
            return;
        }
        while(num > 0){
            digits.push_back(num % base);
            num /= base;
        }
    }

    size_t Size() const {
        return digits.size();
    }

    TBigInt& operator=(const TBigInt &oth){
        digits = oth.digits;
        return *this;
    }

    TBigInt& operator=(TBigInt &&oth){
        digits = std::move(oth.digits);
        return *this;
    }

    static TBigInt Pow(TBigInt bi, TBigInt p){
        TBigInt res(1);
        while(p > 0){
            if(p.digits[0] % 2 == 1){
                res = res * bi;
                p = p - 1;
            }
            bi = bi * bi;
            p = p / 2;
        }
        return  res;
    }

    static TBigInt Pow(TBigInt bi, digit_type p){
        TBigInt res(1);
        while(p > 0){
            if(p % 2 == 1) {
                res = res * bi;
            }
            bi = bi * bi;
            p >>= 1;
        }
        return  res;
    }

private:
    std::vector<digit_type > digits;

    digit_type GetDigit(size_t ind) const {
        return ind < digits.size() ? digits[ind] : 0;
    }

    digit_type& operator[](size_t ind){
        assert(ind < digits.size());
        return  digits[ind];
    }

    void FilterZeros(){
        for(size_t i = digits.size() - 1; i >= 1; i--){
            if(digits[i] != 0){
                break;
            }
            digits.pop_back();
        }
    }

    int Comp(const TBigInt &oth) const {
        size_t size = Size();
        size_t othSize = oth.Size();
        if(size != othSize){
            return size > othSize ? 1 : -1;
        }
        for(size_t i = size - 1; i < size; i--){
            if(digits[i] != oth.digits[i]){
                return digits[i] > oth.digits[i] ? 1 : -1;
            }
        }
        return 0;
    }

    friend TBigInt operator+(const TBigInt &lbi, const TBigInt &rbi);
    friend TBigInt operator-(const TBigInt &lbi, const TBigInt &rbi);
    friend TBigInt operator*(const TBigInt &lbi, const TBigInt &rbi);
    friend TBigInt operator^(const TBigInt &bi, TBigInt p);
    friend TBigInt operator^(const TBigInt &bi, digit_type p);
    friend TBigInt operator/(const TBigInt &lbi, const TBigInt &rbi);
    friend TBigInt operator/(const TBigInt &lbi, const digit_type ri);

    friend bool operator<(const TBigInt &lbi, const TBigInt &rbi);
    friend bool operator>(const TBigInt &lbi, const TBigInt &rbi);
    friend bool operator<=(const TBigInt &lbi, const TBigInt &rbi);
    friend bool operator>=(const TBigInt &lbi, const TBigInt &rbi);
    friend bool operator==(const TBigInt &lbi, const TBigInt &rbi);

    friend std::ostream &operator<<(std::ostream &ostr, const TBigInt &bi);
};

TBigInt operator+(const TBigInt &lbi, const TBigInt &rbi){
    size_t size = std::max(lbi.Size(), rbi.Size());
    TBigInt res(size, size_t(0));

    digit_type rem = 0;
    for(size_t i = 0; i < size; i++){
        res[i] = lbi.GetDigit(i) + rbi.GetDigit(i) + rem;
        rem = res[i] / base;
        res[i] %= base;
    }
    if(rem > 0){
        res.digits.push_back(rem);
    }
    return res;
}

TBigInt operator-(const TBigInt &lbi, const TBigInt &rbi){
    size_t size = std::max(lbi.Size(), rbi.Size());
    TBigInt res(size, size_t(0));

    digit_type rem = 0;
    for(size_t i = 0; i < size; i++){
        res[i] = base + lbi.GetDigit(i) - rbi.GetDigit(i) - rem;
        rem = base <= res[i] ? 0 : 1;
        res[i] %= base;
    }

    res.FilterZeros();
    return res;
}

TBigInt operator*(const TBigInt &lbi, const TBigInt &rbi){
    size_t lSize = lbi.Size();
    size_t rSize = rbi.Size();

    size_t size = lSize + rSize;
    TBigInt res(size, size_t(0));

    for(size_t i = 0; i < rSize; i++){
        if(rbi.GetDigit(i) == 0){
            continue;
        }
        digit_type rem = 0;
        for(size_t j = 0; j < lSize; j++){
            res[i + j] += rbi.GetDigit(i) * lbi.GetDigit(j) + rem;
            rem = res[i + j] / base;
            res[i + j] %= base;
        }
        if(rem > 0){
            res[i + lSize] = rem;
        }
    }

    res.FilterZeros();
    return  res;
}

TBigInt operator^(const TBigInt &bi, TBigInt p){
    return TBigInt::Pow(bi, p);
}

TBigInt operator^(const TBigInt &bi, digit_type p){
    return TBigInt::Pow(bi, p);
}

TBigInt operator/(const TBigInt &lbi, const TBigInt &rbi){
    size_t rSize = rbi.Size();
    if(lbi < rbi) {
        return TBigInt(0);
    }

    TBigInt rem = lbi;
    TBigInt res(0);

    while(rem >= rbi){
        size_t remSize = rem.Size();

        digit_type firstDigit = rem.GetDigit(remSize - 1);
        digit_type secondDigit = remSize > 1 ? rem.GetDigit(remSize - 2) : 0;
        digit_type left = (firstDigit * base + secondDigit) / rbi.GetDigit(rSize - 1);
        digit_type right = 0;
        size_t pow = remSize - rSize - 1;

        TBigInt power = TBigInt(base) ^ pow;
        TBigInt multiplier = rbi * power;
        TBigInt prod;
        while(right <= left){
            digit_type mid = (right + left) / 2;
            prod = TBigInt(mid) * multiplier;
            if(rem < prod){
                left = mid - 1;
            }
            else{
                right = mid + 1;
            }
        }

        rem = rem - TBigInt(left) * multiplier;
        res = res + left * power;
    }
    return res;
}

TBigInt operator/(const TBigInt &lbi, const digit_type ri){
    digit_type rem = 0;
    size_t lSize = lbi.Size();
    TBigInt res(lSize, (size_t)0);

    for(size_t i = lSize - 1; i < lSize; i--){
        digit_type cur = lbi.GetDigit(i) + rem * base;
        res[i] = cur / ri;
        rem = cur % ri;
    }
    res.FilterZeros();
    return res;
}

bool operator<(const TBigInt &lbi, const TBigInt &rbi){
    return rbi.Comp(lbi) > 0;
}

bool operator>(const TBigInt &lbi, const TBigInt &rbi){
    return lbi.Comp(rbi) > 0;
}

bool operator<=(const TBigInt &lbi, const TBigInt &rbi){
    return rbi.Comp(lbi) >= 0;
}

bool operator>=(const TBigInt &lbi, const TBigInt &rbi){
    return lbi.Comp(rbi) >= 0;
}

bool operator==(const TBigInt &lbi, const TBigInt &rbi){
    return rbi.Comp(lbi) == 0;
}

std::ostream& operator<<(std::ostream &ostr, const TBigInt &bi){
    size_t size = bi.digits.size();
    for(size_t i = size - 1; i < size; i--){
        if(i < size - 1){
            ostr << std::setw(baseLen) << std::setfill('0');
        }
        ostr << bi.digits[i];
    }
    return ostr;
}

int main() {    
    std::string str;
    char op;
    while(std::cin >> str){
        TBigInt a(str);
        std::cin >> str;
        TBigInt b(str);
        std::cin >> op;

        switch(op){
            case '+':
                std::cout << a + b << std::endl;
                break;
            case '-':
                if(a < b){
                    std::cout << "Error" << std::endl;
                }
                else{
                    std::cout << a - b << std::endl;
                }
                break;
            case '*':
                std::cout << a * b << std::endl;
                break;
            case '^':
                if(a == TBigInt(0) && b == TBigInt(0)){
                    std::cout << "Error" << std::endl;
                }
                else{
                    std::cout << (a ^ b) << std::endl;
                }
                break;
            case '/':
                if (b == TBigInt(0)){
                    std::cout << "Error" << std::endl;
                }
                else{
                    std::cout << a / b << std::endl;
                }
                break;
            case '>':
                std::cout << (a > b ? "true" : "false") << std::endl;
                break;
            case '<':
                std::cout << (a < b ? "true" : "false") << std::endl;
                break;
            case '=':
                std::cout << (a == b ? "true" : "false") << std::endl;
                break;
        }
    }
    return 0;
}