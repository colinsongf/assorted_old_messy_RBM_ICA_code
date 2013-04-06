#! /bin/bash

DIR='data/upson_rovio_3'
U2_PREFIXES='u2_backward_0_person u2_backward_1 u2_backward_2 u2_backward_3 u2_forward_0_person u2_forward_1 u2_stationary_0_person u2_stationary_1 u2_strafe_r_0 u2_strafe_r_1 u2_strafe_r_2 u2_strafe_r_3 u2_turn_r_0 u2_turn_r_1'
NEW_PREFIXES='u3_all_shapes_tour u3_green_circle_close_1 u3_green_circle_close_2 u3_green_circle_far_1 u3_green_circle_far_2 u3_green_circle_far_high u3_green_star_close_1 u3_green_star_close_2 u3_green_star_far_1 u3_green_star_far_2 u3_green_star_far_high u3_jason_laptop u3_red_circle_close_1 u3_red_circle_close_2 u3_red_circle_far_1 u3_red_circle_far_2 u3_red_circle_far_high u3_red_star_close_1 u3_red_star_close_2 u3_red_star_far_1 u3_red_star_far_2 u3_red_star_far_high'
#NEW_PREFIXES='u3_green_circle_close_1'  # HACK for testing


mkdir -p $DIR/imgfiles

# Uncomment to make U2 images as well
#for prefix in $U2_PREFIXES; do
#    ffmpeg -i $DIR/movfiles/$prefix.mov -f image2 $DIR/imgfiles/$prefix.%05d.png
#done

for prefix in $NEW_PREFIXES; do
    ffmpeg -i $DIR/movfiles/$prefix.mov -f image2 $DIR/imgfiles/$prefix.%05d.big.png
    for file in $DIR/imgfiles/$prefix.*.big.png; do
        newfile=$(echo $file | sed 's/.big//g')
        convert $file -resize 320x240 $newfile
        rm $file
        echo "resized and removed $file"
    done
done
