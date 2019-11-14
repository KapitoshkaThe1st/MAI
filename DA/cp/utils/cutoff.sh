#!/bin/bash

filename=$1
len=$2
lineCount=`wc -l $filename | awk ' {print $1} '`
newLineCount=`echo $lineCount-$len | bc`

tempFile=`date +"temp%s"`

echo "Previous file length: "$lineCount
# echo `wc -l $filename | awk ' {print $1} ' - $len` | bc
echo "New file length: "$newLineCount
head -n $newLineCount $filename > $tempFile
rm -r $filename
mv $tempFile $filename