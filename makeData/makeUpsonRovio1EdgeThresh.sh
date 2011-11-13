#! /bin/bash

DIRIN='data/upson_rovio_1'
DIROUT='data/upson_rovio_1_edge_thresh'

find $DIRIN | egrep 'image-25.*png$' | sort | xargs -n 1 basename | xargs -I {} convert -colorspace gray $DIRIN/{} -edge 2 -threshold 15% $DIROUT/{}

#echo {} | head

# convert -colorspace gray test.png -edge 2 -threshold 128 out.png


exit 0

# To make video
rm -f /tmp/img_*.png
find . | egrep png$ | sort | sed -n '1~1p' | lnseq /tmp/img_%05d.png
ffmpeg -threads 0 -r 30 -i /tmp/img_%05d.png -vcodec libx264 -level 21 -vpre slow -refs 2 -b 2M -bt 4M edgeThreshVideo.mp4
