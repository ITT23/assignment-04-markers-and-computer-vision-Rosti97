[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/I4_dFpC1)

# Task 1:

To start you have to parse some arguments: 
> python3 -in <path_of_file_to_be_transformed> -out <path_of_where_to_save> -width <width_of_end_file> -height <height_of_end_file>
*example*:
>python3 image_extractor.py -in sample_image.jpg -out final.jpg -width 600 -height 400

### Workflow:
- select first the corner that will be the upper left corner in the end
- select following corners clockwise
- after 4 corners, a preview of transformed image will be shown
- this image can be saved with "s" on keyboard
- to restart, press "esc"
- to close window, press "q"