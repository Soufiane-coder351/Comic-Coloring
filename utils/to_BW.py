import os
import csv

path = "./Dataset"

file=open('Dataset.csv',mode='w+')
file.write("Image_path,BW_Image_path\n")

for vol in os.listdir(path):
    
    for img in os.listdir(os.path.join(path,vol,'Colour')):
        # print(os.path.join(path,vol,'Colour',img))
        file.write(os.path.join(path,vol,'Colour',img)+','+os.path.join(path,vol,'BW',img)+'\n')