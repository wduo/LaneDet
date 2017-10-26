import os

'''
def file_name(file_dir):   
    for root, dirs, files in os.walk(file_dir):  
        print(root) #当前目录路径  
        print(dirs) #当前路径下所有子目录  
        print(files) #当前路径下所有非目录子文件  
'''

'''
for root, dirs, files in os.walk("imgs"):
    for name in files:
        print(os.path.join(root, name))
    for name in dirs:
        print(os.path.join(root, name))
'''

print(os.listdir('imgs'))

'''
if __name__ == "__main__":
    file_name('imgs')
'''
