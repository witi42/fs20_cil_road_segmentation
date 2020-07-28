import cv2
import numpy as np
from matplotlib import pyplot as plt

import glob
from PIL import Image

def np_from_files(files: list) -> np.ndarray:
    x = []
    for f in files:
        image = Image.open(f)
        image = np.asarray(image)
    
        x.append(image)
    
    return np.asarray(x)

def subimg(img1,img2):

    img1y=img1.shape[0]
    img1x=img1.shape[1]

    img2y=img2.shape[0]
    img2x=img2.shape[1]

    stopy=img2y-img1y+1
    stopx=img2x-img1x+1

    for x1 in range(0,stopx):
        for y1 in range(0,stopy):
            x2=x1+img1x
            y2=y1+img1y

            pic=img2[y1:y2,x1:x2]
            test=pic==img1

            if test.all():
                return x1, y1

    return False


def find_image(im, tpl):
    im = np.atleast_3d(im)
    tpl = np.atleast_3d(tpl)
    H, W, D = im.shape[:3]
    h, w = tpl.shape[:2]

    # Integral image and template sum per channel
    sat = im.cumsum(1).cumsum(0)
    tplsum = np.array([tpl[:, :, i].sum() for i in range(D)])

    # Calculate lookup table for all the possible windows
    iA, iB, iC, iD = sat[:-h, :-w], sat[:-h, w:], sat[h:, :-w], sat[h:, w:] 
    lookup = iD - iB - iC + iA
    # Possible matches
    possible_match = np.where(np.logical_and.reduce([lookup[..., i] == tplsum[i] for i in range(D)]))

    # Find exact match
    for y, x in zip(*possible_match):
        if np.all(im[y+1:y+h+1, x+1:x+w+1] == tpl):
            return True

    return False

img = cv2.imread('chicago/chicago1_image.png',0)
img2 = img.copy()
template = cv2.imread('cil-road-segmentation-2020/test_images/test_images/test_7.png',0)
w, h = template.shape[::-1]

print(find_image(img,template))



template_images_names = sorted(glob.glob('cil-road-segmentation-2020/test_images/test_images/*.png'))
dataset_images_names = sorted(glob.glob('chicago/*_image.png'))

print(len(template_images_names))
print(len(dataset_images_names))


template_images = np_from_files(template_images_names)
dataset_images = np_from_files(dataset_images_names)

'''
with open('template_images.npy', 'wb') as f:
    np.save(f, template_images)

with open('dataset_images.npy', 'wb') as f:
    np.save(f, dataset_images_names)

#print(template_images)
#print(dataset_images)


#template_images = np.load('template_images.npy')

#dataset_images = np.load('dataset_images.npy')
'''
remove_image = {}

for temp, temp_name in zip(template_images, template_images_names): 
    for img, img_name in zip(dataset_images, dataset_images_names):
        if(find_image(img,temp)):
            remove_image.append({temp_name, img_name})


print(remove_image)




'''
result = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
StartButtonLocation = np.unravel_index(result.argmax(),result.shape)

print(StartButtonLocation)





# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()
'''