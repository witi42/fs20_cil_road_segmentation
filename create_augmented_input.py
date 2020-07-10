import os
from skimage.io import imsave
from skimage.util import img_as_ubyte
from skimage import filters
from skimage.color import rgb2gray
from skimage.color import gray2rgb
from skimage.color import rgb2hsv
from skimage.color import hsv2rgb



def create_augmented_training_data(size = None, clear_existing = False, use_grayscale = False, use_blur = False, use_saturated = False, use_desaturated = False):
    path_images = "input/training/augmented/images/"
    path_groundtruth = "input/training/augmented/groundtruth/"
    os.makedirs(path_images, exist_ok = True)
    os.makedirs(path_groundtruth, exist_ok = True)

    if clear_existing:
        clear_augmented_data()

    x, y = get_training_data(rotate = False, size = size)

    for i in range(len(x)):
        print("\r" + str(i) + "/" + str(len(x)), end="")
    
        img = img_as_ubyte(x[i])
        img_gt = img_as_ubyte(y[i] * 1.0)

        if use_grayscale:
            img_gs = img_as_ubyte(gray2rgb(rgb2gray(img)))
            filename_gs = "gs_" + str(i) + ".png"
            imsave(path_images + filename_gs, img_gs)
            imsave(path_groundtruth + filename_gs, img_gt)
    
        if use_blur:
            img_blur = img_as_ubyte(filters.gaussian(img, sigma = 1.5, multichannel = True))
            filename_blur = "blur_" + str(i) + ".png"  
            imsave(path_images + filename_blur, img_blur)
            imsave(path_groundtruth + filename_blur, img_gt)
    
        if use_saturated:
            img_sat = rgb2hsv(img)
            img_sat[:, :, 1] = (1 - (1 - img_sat[:, :, 1]) ** 2)
            img_sat = img_as_ubyte(hsv2rgb(img_sat))
            filename_sat = "sat_" + str(i) + ".png"  
            imsave(path_images + filename_sat, img_sat)
            imsave(path_groundtruth + filename_sat, img_gt)

        if use_desaturated:
            img_desat = rgb2hsv(img)
            img_desat[:, :, 1] = (1 - (1 - img_desat[:, :, 1]) ** 0.5)
            img_desat = img_as_ubyte(hsv2rgb(img_desat))
            filename_desat = "desat_" + str(i) + ".png"  
            imsave(path_images + filename_desat, img_desat)
            imsave(path_groundtruth + filename_desat, img_gt)

  
    print("\rdone       ")



def clear_augmented_data():
    path_images = "input/training/augmented/images/"
    path_groundtruth = "input/training/augmented/groundtruth/"
    for path in [path_images, path_groundtruth]:
        files = glob.glob(path + '*.png')
        for f in files:
            os.remove(f)





if __name__ == "__main__":
    USE_AUGMENTED_DATA = True

    x, y = get_training_data(rotate = True, flip = True, size = DATA_SIZE)
    x_test, x_test_names = get_test_data(size = DATA_SIZE)

    if USE_AUGMENTED_DATA:
        create_augmented_training_data(DATA_SIZE, clear_existing = True,
                                       use_grayscale = False,
                                       use_blur = False,
                                       use_saturated = False,
                                       use_desaturated = True)
        x_aug, y_aug = get_augmented_training_data(rotate = True, flip = True, size = DATA_SIZE)

        print((x.shape, x_aug.shape))
        print((y.shape, y_aug.shape))
    
        if len(x_aug.shape) == 4:
            x = np.concatenate((x, x_aug), axis = 0)
            y = np.concatenate((y, y_aug), axis = 0)
