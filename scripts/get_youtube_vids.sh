#!/bin/bash

input="videos.txt"
download_dir="."

urls=()
start_times=()
end_times=()

while IFS= read -r line
do
    IFS=' '   
    read -ra ADDR <<<"$line"   
    urls+=(${ADDR[0]})
    start_times+=(${ADDR[1]})
    end_times+=(${ADDR[2]})
done < $input


n=${#urls[@]}
for ((i=1;i<=n;i++)); do

    echo "Downloading video $i of $n"

    start_secs=$((${start_times[$i]}%60))
    start_mins=$((${start_times[$i]}/60))
    start_hours=$((${start_times[$i]}/3600))

    end_secs=$((${end_times[$i]}%60))
    end_mins=$((${end_times[$i]}/60))
    end_hours=$((${end_times[$i]}/3600))

    start_time="$start_hours:$start_mins:$start_secs"
    end_time="$end_hours:$end_mins:$end_secs"

    video_path="${download_dir}/${i}.mp4"

    ffmpeg -hide_banner -loglevel error -ss ${start_time} -to ${end_time} -i "$(youtube-dl -f best --get-url ${urls[$i]})" -c:v copy -c:a copy ${video_path}

done