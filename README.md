# fs20_cil_road_segmentation - CutTheRoad- Team fubar
https://gitlab.ethz.ch/robinw/fs20_cil_road_segmentation

## Dependencies and Environment
Our conda-environment (Linux) is provided in *environment.yml*. Create the environment (*conda env create -f environment.yml -n RS && conda activate RS*) of install the packages in *requirements.txt*.

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
The dataset splits (train, validation) for different levels of augmentation can be create by running *python create_augmented_data_small.py*. This will create the folders *input/ds/* and *input/ds_aug_small/* containing datests with the augmented images.

(If you want to run training with *itermediate* or *large* augmentation run *python create_augmented_data.py*. Note however, this requires about 60GB of free disk storage)

# Train Model and create Submission
To recreate our best results, look in section **Training with Augmentation (BEST RESULTS)** (*python train_for_submission_augmentation_cnn.py*).
For all of these files submissions will automatically be created and can be found in *submission_csv/*. 

## Crossvalidation on Original Data

5-fold cross-validation can be run on the different models by executing the follwing files

python
* *train_val_unet.py* - unet with all losses
* *train_val_cnn.py* - deep unet with all losses
* *train_val_sdfbasic.py* - novel sdf models with unet
* *train_val_sdfcnn.py* - novel sdf models with deep unet
* *train_val_sdf_losses.py* - novel sdf models with deep unet and additional dice and crossentropy loss

## Training on Chicago
For the scripts of this section, the Original data is split into 70 images for training and 30 for validation. The Chicago data is added to the training data.

### Training wihtout Augmentation
**Novel Solution:**
python
* *train_for_submission_novel.py*
* *train_for_submission_novel_dice.py*

**Unet with different losses** (this requires up to 32GB RAM, files from training with augmentation use an iterator instead so 16GB or less is sufficient)
python
* *train_for_submission.py*

### Training with Augmentation and Postprocessing (BEST RESULTS)
python
* *train_for_submission_augmentation_cnn.py* - trains with small level of augmentation. Remove the comments on the the *# medium augmentation* or the *# large augmentation* sections in order to train with those levels of augmentation.
One submission file with and one submission file without postprocessing is automatically created in *submission_csv/*.



## Ensemble Model
To combine multiple solutions and average them, copy the submission files you want to bag (the `.csv`s that you would submit to kaggle) and run `python combine_submissions.py`. Additionally, you can set the threshhold which is used to binarize the mean score in *combine_submissions.py*. The output will then be saved in the director *ensemble-test*, along with visualisations of the predicted mask overlayed on the test images.

## Postprocessing
Given an image and its predicted labels, you can then use the crf function in the crf modules to perform post-processing as follows:
`new_predictions = crf.crf(image, prediction)` 
and optionally change the hyperparameters using the provided parameters.
*train_for_submission_augmentation_cnn.py* automatically creates a submission with postprocessing.