#!/bin/sh

if [ $# != 1 ]; then
    echo "usage: $0 filename"
    exit 1
fi

filename=$1

sed -i '' 's,\(!.*(\).*/\(.*\)),\1https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/\2?raw=true),' $filename 
