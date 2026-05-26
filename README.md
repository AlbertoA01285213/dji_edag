# dji_edag
 
alberto@Alberto:~/Documents/dji_edag/slam/video_2$ ffmpeg -i video.mp4 -f image2 ~/Documents/dji_edag/slam/video_2/%05d.png


cd ~/Documents/dji_edag/slam/video
ls frame_*.png | awk '{print NR*0.0333, $1}' > rgb.txt

./Examples/Monocular/mono_tum   Vocabulary/ORBvoc.txt   Examples/Monocular/laptop.yaml   /home/alberto/Documents/dji_edag/slam/video_2
