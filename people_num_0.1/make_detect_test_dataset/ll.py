import os

path = 'F:/person_count_testset/tangyanzhongxin/smwy/'
for name in os.listdir(path):
    new_name = name[0:15]+name[16:]
    print(new_name)
    os.rename(path+name,path+new_name)