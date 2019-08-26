#!/bin/bash

cd data

# download COCO images
mkdir images
cd images
wget -c https://pjreddie.com/media/files/train2014.zip
wget -c https://pjreddie.com/media/files/val2014.zip
unzip -q -j train2014.zip
unzip -q -j val2014.zip
cd ..

# download COCO annotations
wget -c https://pjreddie.com/media/files/coco/labels.tgz
mkdir labels
cd labels
tar xzf ../labels.tgz --strip=2
cd ..
wget -c https://pjreddie.com/media/files/instances_train-val2014.zip
unzip -q instances_train-val2014.zip

# create images list
wget -c https://pjreddie.com/media/files/coco/trainvalno5k.part
paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt
sed -i "s/train2014\///g" trainvalno5k.txt
sed -i "s/val2014\///g" trainvalno5k.txt
sed -i "/COCO_train2014_000000167126.jpg/d" trainvalno5k.txt  # remove corrupted instance
wget -c https://pjreddie.com/media/files/coco/5k.part
paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt
sed -i "s/val2014\///g" 5k.txt

# download COCO classes names
wget -c https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

cd ..

