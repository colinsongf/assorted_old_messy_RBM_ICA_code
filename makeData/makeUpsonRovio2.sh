#! /bin/bash

DIR='data/upson_rovio_2'
PREFIXES='u2_backward_0_person u2_backward_1 u2_backward_2 u2_backward_3 u2_forward_0_person u2_forward_1 u2_stationary_0_person u2_stationary_1 u2_strafe_r_0 u2_strafe_r_1 u2_strafe_r_2 u2_strafe_r_3 u2_turn_r_0 u2_turn_r_1'

mkdir $DIR/imgfiles

for prefix in $PREFIXES; do
    ffmpeg -i $DIR/movfiles/$prefix.mov -f image2 $DIR/imgfiles/$prefix.%05d.png
done
