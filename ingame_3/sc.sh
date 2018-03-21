#!/bin/bash

shopt -s extglob
shopt -s nocaseglob

IMAGE_TYPES='npy'
IFS=$'\n' dirlist=(`find "$PWD" -type d`)

counter=0
for dir in "${dirlist[@]}"; do
    cd "$dir"
    ls *.+($IMAGE_TYPES) > /dev/null 2>&1 || continue

    for file in *.+($IMAGE_TYPES); do
        printf -v newname "../%.3d.%s" $((counter += 1)) "npy"
	#printf "$newname"
        mv --verbose "$file" "$newname"
    done
done
