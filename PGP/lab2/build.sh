#!/bin/bash

grid_dims=(1 32 128 256 512 1024)
block_dims=(32 64 128 256 512 1024)

min_time=100000000000
res=0

if [ -f "stats" ]; then
    rm stats
fi

CFLAGS=(-Werror cross-execution-space-call -lm)
BIN=lab2

echo -n "---," >> stats
for bd in "${block_dims[@]}"; do
    if [[ $bd != ${block_dims[0]} ]]; then
        echo -n "," >> stats
    fi
    echo -n "$bd блок" >> stats
done
echo >> stats

for gd in "${grid_dims[@]}"; do
    echo -n "$gd грид," >> stats

    for bd in "${block_dims[@]}"; do
        echo "<$gd, $bd>"
        exe_name=$BIN-$gd-$bd
        if [[ $1 == "recompile" ]]; then
            echo "nvcc "${CFLAGS[@]}" --define-macro BENCHMARK --define-macro GRID=$gd --define-macro BLOCK=$bd -o $exe_name main.cu"
            nvcc "${CFLAGS[@]}" --define-macro BENCHMARK --define-macro GRID=$gd --define-macro BLOCK=$bd -o $exe_name main.cu
        fi

        res=$(./$exe_name < $2)

        if (( $(awk "BEGIN {print $res < $min_time}") )); then
            params="<$gd, $bd>"
            min_time=$res
        fi
        # echo -n "<$gd, $bd> $res;" >> stats
        if [[ $bd != ${block_dims[0]} ]]; then
            echo -n "," >> stats
        fi
        echo -n "$res" >> stats
    done
    echo >> stats
done

echo "best: $params with time $min_time" > best