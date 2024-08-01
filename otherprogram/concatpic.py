import os
import os.path
import cv2
import numpy as np

def fixpic(images):
    new_images = []
    height, width, channels = images[0].shape
    #sub_image_height = height
    sub_image_width = width // 4
    
    for image in images:
        image2 = image[:, sub_image_width:2 * sub_image_width, :]
        image3 = image[:, 2 * sub_image_width:3 * sub_image_width, :]
        image4 = image[:, 3 * sub_image_width:4 * sub_image_width, :]

        new_image = np.concatenate((image2, image4, image3), axis=1)
        new_images.append(new_image)
        
    return new_images


def concatenate_images_v1xv2(images, v1, v2):
    #if len(images) != v1*v2:
    #    raise ValueError("The number of images must be 60.")
    
    # Assuming all images have the same dimensions
    height, width, channels = images[0].shape
    combined_image = np.zeros((v2 * height, v1 * width, channels), dtype=np.uint8)
    
    for idx, image in enumerate(images):
        row = idx % v2
        col = idx // v2
        y = row * height
        x = col * width
        combined_image[y:y + height, x:x + width] = image
    
    return combined_image

path = '/home/hhc102u/SDT/Generated/2sets_test_wm-all+wc_redo(187999)/test/'
#path = '/home/hhc102u/SDT/Generated/2sets_test_June_bestcheck'

fbname= '_湖_sdt_gt_g1.png'

files = []

for i in range(60):
    dirnum=str(i)
    dirpath = os.path.join(path, dirnum)
    fname = dirnum+fbname
    
    file = os.path.join(dirpath, fname)
    if os.path.exists(file):
        files.append(file)

print(len(files))    
    
images = []
for fpath in files:
    if os.path.exists(fpath):
        image = cv2.imread(fpath)
        if image is not None:
            images.append(image)
        else:
            print(f"Failed to load image at {fpath}")
    else:
        print(f"Path does not exist: {fpath}")    

images = fixpic(images)

img = concatenate_images_v1xv2(images, 4, 15)
imgpath = os.path.join(path, '湖all60.png')
cv2.imwrite(imgpath, img)