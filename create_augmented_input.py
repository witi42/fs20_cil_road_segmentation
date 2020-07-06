import os
from skimage.io import imsave
from skimage.util import img_as_ubyte
from skimage import filters
from skimage.color import rgb2gray
from skimage.color import gray2rgb
from skimage.color import rgb2hsv
from skimage.color import hsv2rgb



def create_augmented_training_data():
    path_images = "input/training/augmented/images/"
    path_groundtruth = "input/training/augmented/groundtruth/"
    os.makedirs(path_images, exist_ok = True)
    os.makedirs(path_groundtruth, exist_ok = True)
    
    x, y = get_training_data(rotate = False, flip = False, size = (400, 400))
    
    # If some augmented data is not wanted anymore, but has already been created, it's necessary to
    #   delete it from input/training/augmented/, as get_augmented_training_data() would still load it
    use_grayscale = True
    use_blur = True
    use_saturated = True
    
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

  
  print("\rdone       ")





if __name__ == "__main__":
    create_augmented_training_data()
    x_aug, y_aug = get_augmented_training_data(rotate = False, size = (224, 224))

    print(x_aug.shape)
    print(y_aug.shape)

    print(y_aug[1, 100:110,100:110])