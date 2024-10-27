import os
import cv2

path = "./Dataset"

file = open("Dataset.csv", mode="w+")
file.write("Image_path,BW_Image_path\n")


def add_paths_image(path, file):
    ''' add paths to dataset.csv and create the bw image '''
    for vol in os.listdir(path):
        # create the BW folders
        bw_path = os.path.join(path, vol, "BW")
        if not os.path.exists(bw_path):
            os.mkdir(bw_path)

        for img in os.listdir(os.path.join(path, vol, "Colour")):
            path_color = os.path.join(path, vol, "Colour", img)
            path_bw = os.path.join(path, vol, "BW", img)

            file.write(path_color + "," + path_bw + "\n")
            create_BW_image(path_color, path_bw)


def create_BW_image(path_color, path_bw):
    image = cv2.imread(path_color)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path_bw, gray_image)


add_paths_image(path, file)
