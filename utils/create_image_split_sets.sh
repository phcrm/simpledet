#! /bin/zsh

IMG_PATH=data/src/mahd/JPEGImages
SPLIT_PATH=data/src/mahd/ImageSets/Main

files=($(ls $IMG_PATH | grep ".jpg" | sed s/.jpg// | shuf))

# 80/20 split
num=${#files[@]}
train=$(($num*0.8))

for f in ${files[*]:0:$train}; do
    echo $f >> $SPLIT_PATH/train.txt
done

for f in ${files[*]:$train:$num}; do
    echo $f >> $SPLIT_PATH/test.txt
done