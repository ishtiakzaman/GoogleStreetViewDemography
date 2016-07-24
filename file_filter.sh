#!/bin/bash

while read -r line
do
  name=${line%%\ *}
  if [ -f  data2/$name ]
  then
    echo $line
  fi
done < "data2.txt"