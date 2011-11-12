#! /bin/bash

DIR='data/upson_rovio_1'

ffmpeg -i $DIR/yos_s2534.mov -f image2 $DIR/image-2534-%05d.png
ffmpeg -i $DIR/yos_s2535.mov -f image2 $DIR/image-2535-%05d.png
ffmpeg -i $DIR/yos_s2545.mov -f image2 $DIR/image-2545-%05d.png
