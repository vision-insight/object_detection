#!/bin/sh

#scp -P 61019 -r /data/object_detection/yolov3/result/* vola@39.104.145.211:~/result/

source_path="/data/object_detection/people_count/output/"
dest_path="~/Desktop/result/"
echo $path
rsync -avPrtzc --delete --timeout 60 -e "ssh -p 61019" $source_path  vola@39.104.145.211:$dest_path

