# fs20_cil_road_segmentation - CutTheRoad- Team fubar
https://gitlab.ethz.ch/robinw/fs20_cil_road_segmentation

## Dependencies and Environment
Install the conda environ

## Prepare Data
### Download
All data must be stored and will be loaded from *input/*
Download and extract the dataset from kaggle. Place the images in *input/original/image/* and the labels with the same names in *input/original/label/*. Place the test images in *input/test_images/*

Download and unpack the chicago dataset from [GoogleDrive](https://drive.google.com/file/d/1ZcZdUjGD8M0XOt7ssgMT6EXCNtAkLz7K/view?usp=sharing). Place the unpacked *chicago/* folder in *input/*. Your folder should look like this:
* input/
    * original/
        * image/
        * label/
    * chicago/
        * image/
        * label/
    * test_images/

(Instead of downloading the processed images from drive you can also download the original *chicago.zip* from [zenodo](https://zenodo.org/record/1154821#.XyQB2nUzZhl).Unzip all contained images directly into *generate_data/chicago/* and then process them by running *python generate_data/process_original_chicago.py* from within the project's root folder)


### Create Splits
Now create the dataset splits (train, validation) for different levels of augmentation can be create by running *python create_augmented_data_small.py*. This will create the folders *input/ds/* and *input/ds_aug_small/* containing datests with the augmented images.

(If you want to run training with *itermediate* or *large* augmentation run *python create_augmented_data.py*. Note however, this requires about 60GB of free disk storage)

# Train Model and create Submission

## Crossvalidation on Original Data

5-fold cross-validation can be run on the different models by executing the follwing files

* *train_val_unet.py* - unet with all losses
* *train_val_cnn.py* - deep unet with all losses
* *train_val_sdfbasic.py* - novel sdf models with unet
* *train_val_sdfcnn.py* - novel sdf models with deep unet
* *train_val_sdf_losses.py* - novel sdf models with deep unet and additional dice and crossentropy loss


## Ensemble Model
To combine multiple solutions and average them, copy the submission files you want to bag (the *.csv*s that you would submit to kaggle) and run *python combine_submissions.py*. Additionally, you can set the threshhold which is used to binarize the mean score in *combine_submissions.py*. The output will then be saved in the director *ensemble-test*, along with visualisations of the predicted mask overlayed on the test images.


## 