import os

root = '/media/D/test_data/person_count/'
for dir_name in os.listdir(root):

    xmlfilepath =root+dir_name+'/test_labels/'
    txtsavepath = 'ImageSets/'
    total_xml = os.listdir(xmlfilepath)

    num=len(total_xml)
    list=range(num)

    ftest = open(txtsavepath+'test.txt', 'a')

    for i in list:
        name=total_xml[i][:-4]+'\n'
        ftest.write(name)
ftest.close()
