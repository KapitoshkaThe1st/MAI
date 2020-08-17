#ifndef BLAKE_H
#define BLAKE_H

#include <iostream>
#include <fstream>
#include <algorithm>

#include <limits.h>
#include <stdint.h>

#include <bitset>

#define UINT_BITS (sizeof(unsigned int) * 8)
#define UINT_BITS_M1 (UINT_BITS - 1)

unsigned int iv[8] = {
    0x6A09E667, 0xBB67AE85,
    0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C,
    0x1F83D9AB, 0x5BE0CD19};

unsigned int c[16] = {
    0x243F6A88, 0x85A308D3,
    0x13198A2E, 0x03707344,
    0xA4093822, 0x299F31D0,
    0x082EFA98, 0xEC4E6C89,
    0x452821E6, 0x38D01377,
    0xBE5466CF, 0x34E90C6C,
    0xC0AC29B7, 0xC97C50DD,
    0x3F84D5B5, 0xB5470917};

unsigned char sgm[160] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3,
    11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4,
    7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8,
    9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13,
    2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9,
    12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11,
    13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10,
    6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5,
    10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0};

unsigned char sigma(unsigned int r, unsigned int i) {
    return sgm[r * 16 + i];
}

unsigned int cons(unsigned int i) {
    return c[i];
}

#define P2_32 4294967296
#define BYTES_TO_READ (sizeof(unsigned int) * 16)

unsigned int summod2_32(unsigned int a, unsigned int b) {
    return ((unsigned long int)a + b) % P2_32;
}

uint32_t rotleft(uint32_t value, unsigned int count) {
    const unsigned int mask = CHAR_BIT * sizeof(value) - 1;
    count &= mask;
    return (value << count) | (value >> (-count & mask));
}

uint32_t rotright(uint32_t value, unsigned int count) {
    const unsigned int mask = CHAR_BIT * sizeof(value) - 1;
    count &= mask;
    return (value >> count) | (value << (-count & mask));
}

void G(unsigned int *m, unsigned int &a, unsigned int &b, unsigned int &c, unsigned int &d, unsigned int r, unsigned int i) {
    unsigned char s = sigma(r % 10, 2 * i), s1 = sigma(r % 10, 2 * i + 1);
    a = summod2_32(a, summod2_32(b, m[s] ^ cons(s1)));
    d = rotright(d ^ a, 16);
    c = summod2_32(c, d);
    b = rotright(b ^ c, 12);
    a = summod2_32(a, summod2_32(b, m[s1] ^ cons(s)));
    d = rotright(d ^ a, 8);
    c = summod2_32(c, d);
    b = rotright(b ^ c, 7);
}

void round(unsigned int *m, unsigned int *v, unsigned int r) {
    G(m, v[0], v[4], v[8], v[12], r, 0);
    G(m, v[1], v[5], v[9], v[13], r, 1);
    G(m, v[2], v[6], v[10], v[14], r, 2);
    G(m, v[3], v[7], v[11], v[15], r, 3);
    G(m, v[0], v[5], v[10], v[15], r, 4);
    G(m, v[1], v[6], v[11], v[12], r, 5);
    G(m, v[2], v[7], v[8], v[13], r, 6);
    G(m, v[3], v[4], v[9], v[14], r, 7);
}

void compress(unsigned int *h, unsigned int *m, unsigned int *s, unsigned int *t, unsigned int *res, unsigned int round_count=14) {
    unsigned int v[16];

    for (int i = 0; i < 8; ++i)
        v[i] = h[i];
    for (int i = 0; i < 4; ++i) {
        v[i + 8] = s[i] ^ c[i];
    }
    v[12] = t[0] ^ c[4];
    v[13] = t[0] ^ c[5];
    v[14] = t[1] ^ c[6];
    v[15] = t[1] ^ c[7];

    for (unsigned int i = 0; i < round_count; ++i) {
        round(m, v, i);
    }

    for (int i = 0; i < 8; ++i) {
        res[i] = h[i] ^ s[i % 4] ^ v[i] ^ v[i + 8];
    }
}

union block_t{
    unsigned char c[64];
    unsigned int i[16];
    unsigned long int li[8];
};

void blake_hash(std::ifstream &in, unsigned int *res, unsigned int round_count=14){
    unsigned int h0[8], h1[8];

    for (int i = 0; i < 8; ++i)
        h0[i] = iv[i];

    unsigned int s[4] = {0, 0, 0, 0};
    unsigned long int t = 0;
    block_t m;

    unsigned int k = 0;
    while (true) {
        if (!in)
            break;

        in.read((char *)m.i, BYTES_TO_READ);
        unsigned int count = in.gcount();

        t += count * 8;
        if (count != BYTES_TO_READ) {
            m.c[count] = (1 << 7);
            unsigned int i = count + 1;
            for (; i < BYTES_TO_READ - 9; ++i) {
                m.c[i] = 0;
            }
            m.c[i] = 1;

            for (int j = 0; j < 14; ++j) {
                unsigned char *b = (unsigned char *)&m.i[j];
                std::swap(b[0], b[3]);
                std::swap(b[1], b[2]);
            }

            m.li[7] = t;
            std::swap(m.i[14], m.i[15]);
        }
        else{
            for (int j = 0; j < 16; ++j) {
                unsigned char *b = (unsigned char *)&m.i[j];
                std::swap(b[0], b[3]);
                std::swap(b[1], b[2]);
            }
        }
        
        if (k++ % 2 == 0) {
            compress(h0, m.i, s, (unsigned int *)&t, h1, round_count);
        } else {
            compress(h1, m.i, s, (unsigned int *)&t, h0, round_count);
        }
    }

    if (k % 2) {
        for (int i = 0; i < 8; ++i) {
            res[i] = h1[i];
        }
    } else {
        for (int i = 0; i < 8; ++i) {
            res[i] = h0[i];
        }
    }
}

void blake_hash(const char *data, unsigned int n, unsigned int *res, unsigned int round_count=14) {
    unsigned int h0[8], h1[8];

    for (int i = 0; i < 8; ++i)
        h0[i] = iv[i];

    unsigned int s[4] = {0, 0, 0, 0};
    unsigned long int t = 0;
    block_t m;

    unsigned int k = 0;
    while (k * BYTES_TO_READ < n) {
        unsigned int count = 0;

        for(; count < BYTES_TO_READ; ++count){
            unsigned int j = k * BYTES_TO_READ + count;
            if(j >= n)
                break;
            m.c[count] = data[j];
        }

        t += count * 8;
        if (count != BYTES_TO_READ) {
            m.c[count] = (1 << 7);
            unsigned int i = count + 1;
            for (; i < BYTES_TO_READ - 9; ++i) {
                m.c[i] = 0;
            }
            m.c[i] = 1;

            for (int j = 0; j < 14; ++j) {
                unsigned char *b = (unsigned char *)&m.i[j];
                std::swap(b[0], b[3]);
                std::swap(b[1], b[2]);
            }

            ++i;

            m.li[7] = t;
            std::swap(m.i[14], m.i[15]);
        }
        else{
            for (int j = 0; j < 16; ++j) {
                unsigned char *b = (unsigned char *)&m.i[j];
                std::swap(b[0], b[3]);
                std::swap(b[1], b[2]);
            }
        }

        if (k++ % 2 == 0) {
            compress(h0, m.i, s, (unsigned int *)&t, h1, round_count);
        } else {
            compress(h1, m.i, s, (unsigned int *)&t, h0, round_count);
        }
    }

    if (k % 2) {
        for (int i = 0; i < 8; ++i) {
            res[i] = h1[i];
        }
    } else {
        for (int i = 0; i < 8; ++i) {
            res[i] = h0[i];
        }
    }
}

#endif