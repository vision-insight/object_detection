import xml.etree.ElementTree as ET
import os


sets = [('2019', 'test')]
classes = ["person"]

root = '/media/D/test_data/person_count/'
save_label_path = 'label'
ImageSets_path = 'ImageSets'

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id,root):
    for dir_name in os.listdir(root):
        if dir_name == image_id.split('_')[0]:
            in_file = open(root+dir_name+'/test_labels/%s.xml'%(image_id))
            out_file = open(save_label_path+'/%s.txt'%(image_id), 'w')
            tree=ET.parse(in_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 1:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert((w,h), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wwd = os.getcwd()

print(wwd)
wd = wwd.replace("\\","/")
print(wd)

for year, image_set in sets:
    if not os.path.exists(save_label_path):
        os.makedirs(save_label_path)
    image_ids = open(ImageSets_path+'/%s.txt'%(image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    c = 1
    for image_id in image_ids:
        for dir_name in os.listdir(root):
            if dir_name == image_id.split('_')[0]:
                list_file.write(root+dir_name+'/test_imgs/%s.jpg\n'%(image_id))
                convert_annotation(image_id,root)
                print(c)
                c += 1
    list_file.close()
