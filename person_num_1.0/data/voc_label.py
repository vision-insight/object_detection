import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets=[('2019', 'train'), ('2019', 'val'), ('2019', 'test')]

classes = ["person"]

path = '/media/D/train_data/person_count/'

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id):
    for uptown_name in os.listdir(path+'VOCdevkit/VOC2019/Annotations'):
        print(uptown_name,image_id.split('/')[0])
        if uptown_name == image_id.split('/')[0]:
            in_file = open(path+'VOCdevkit/VOC2019/Annotations/'+uptown_name+'/%s.xml'%(image_id.split('/')[1]))
            out_file = open(path+'VOCdevkit/VOC2019/labels/'+uptown_name+'/%s.txt'%(image_id.split('/')[1]), 'w')
            tree=ET.parse(in_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult)==1:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert((w,h), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()
c = 0
for year, image_set in sets:
    if not os.path.exists(path+'VOCdevkit/VOC%s/labels/'%(year)):
        os.mkdir(path+'VOCdevkit/VOC%s/labels/'%(year))
    image_ids = open(path+'VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for uptown_name in os.listdir(path+'VOCdevkit/VOC2019/Annotations'):
        for image_id in image_ids:
            if uptown_name == image_id.split('/')[0]:
                list_file.write(path+'VOCdevkit/VOC2019/JPEGImages/'+uptown_name+'/%s.jpg\n'%(image_id.split('/')[1]))
                c += 1
                print(c)
                convert_annotation(image_id)
    list_file.close()

#os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
#os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")

