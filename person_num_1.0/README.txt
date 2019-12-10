This project is forked from the repro of https://github.com/AlexeyAB/darknet

[compile on Linux(cmake)]
1  cd darknet-master
2  mkdir build-release
3  cd build-release
4  cmake ..
5  make
6  make install





[train]
image_path : /media/D/train_data/person_count/VOCdevkit/VOC2019/JPEGImages
xml_path   : /media/D/train_data/person_count/VOCdevkit/VOC2019/Annotations
run data/voc_label.py

./darknet detector train cfg/person_num.data cfg/yolov3.cfg pre_train_model.weights

# if you want to train on multi-GPU
./darknet detector train cfg/person_num.data cfg/yolov3.cfg per_train_model.weigths -gpus 0,1,2,3

# hyper-parammeter setting
(cfg/yolov3.cfg)
batch = 64
subdivisions = 16
max_batches = 6000 (if you train for 3 classes) (classes * 2000 but not less than 4000)
classes = 1 (in each of 3 [yolo] layers)
filters = 18 ((classes + 5 ) * 3)  in the 3 [convolutional] before each [yolo] layer

(cfg/person_num.data)
classes = 1 (the number of your classes)
train = data/2019/train.txt
valid = data/2019/val.txt
names = data/person_num.names
backup = backup

(data/person_num.names)
person

After training is complete -get result yolov3_last.weights from path backup


[detect]
./darknet detector test cfg/person_num.data cfg/yolov3.cfg person_num.weights -thresh 0.3

# if you want to output coordinates
./darknet detector test cfg/person_num.data cfg/yolov3.cfg person_num.weights
  -thresh 0.3 -ext_output image_path   
