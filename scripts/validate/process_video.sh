#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 path_to_video"
    exit 1
fi

video_path=$1
desired_width=1280
black_rect_height=360
board_size=256
font_size=20

dir_name=$(dirname "$video_path")
base_name=$(basename "$video_path")
file_name="${base_name%.*}"

output_dir="$dir_name/$file_name"
mkdir -p "$output_dir"

ffmpeg -i "$video_path" -q:v 1 -pix_fmt rgb24 "$output_dir/%05d.png"

python scripts/validate/validate_board_recognition.py -i "$output_dir"

marked_dir="${output_dir}_marked"
boards_dir="${output_dir}_boards"
extracted_dir="${output_dir}_extracted"

fps=$(ffmpeg -i "$video_path" 2>&1 | sed -n "s/.*, \(.*\) fps.*/\1/p")

combined_dir="${output_dir}_combined"
mkdir -p "$combined_dir"

x_offset_left=$(( (desired_width / 4) - (board_size / 2) ))
x_offset_right=$(( (desired_width * 3 / 4) - (board_size / 2) ))
board_height=$(( board_size + font_size ))

for img in "$output_dir"/*.png; do
    img_name=$(basename "$img")
    marked_img="$marked_dir/$img_name"
    board_img="$boards_dir/$img_name"
    extracted_img="$extracted_dir/$img_name"
    combined_img="$combined_dir/$img_name"
    temp_image="$combined_dir/temp.jpg"

    convert "$marked_img" -resize "${desired_width}x" "$temp_image"
    resized_height=$(identify -format "%h" "$temp_image")
    total_height=$((resized_height + black_rect_height))
    y_offset=$(( resized_height + (black_rect_height - board_size) / 2 ))

    convert -size ${desired_width}x${total_height} xc:black \
        \( "$temp_image" -geometry +0+0 \) -composite \
        "$combined_img"

    convert -size ${board_size}x${board_height} xc:black \
        \( "$extracted_img" -resize ${board_size}x -gravity south -geometry +0+0 \) -composite \
        "$temp_image"
    convert "$temp_image" -gravity north -pointsize $font_size -fill white -annotate +0+0 "Extracted board:" "$temp_image"
    
    convert "$combined_img" \
        \( "$temp_image" -geometry +${x_offset_left}+${y_offset} \) -composite \
        "$combined_img"

    convert -size ${board_size}x${board_height} xc:black \
        \( "$board_img" -resize ${board_size}x -gravity south -geometry +0+0 \) -composite \
        "$temp_image"
    convert "$temp_image" -gravity north -pointsize $font_size -fill white -annotate +0+0 "Recognized board:" "$temp_image"

    convert "$combined_img" \
        \( "$temp_image" -geometry +${x_offset_right}+${y_offset} \) -composite \
        "$combined_img"

    rm "$temp_image"
done

ffmpeg -framerate "$fps" -i "$combined_dir/%05d.png" -c:v libx264 -pix_fmt yuv420p "${output_dir}_processed.mp4"

# rm -rf "$combined_dir"

echo "New video created at ${output_dir}_processed.mp4"
