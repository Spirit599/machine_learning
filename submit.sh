#!/bin/sh

git add .

echo "input the information of the submit:"
read info

git commit -m $info

git push
