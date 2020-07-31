# fs20_cil_road_segmentation - CutTheRoad- Team fubar

## Dependencies and Environment
Install the conda environ

## Prepare Data
### Download
All data must be stored and will be loaded from *input/*
Download and extract the dataset from kaggle. Place the images in *input/original/image/* and the labels with the same names in *input/original/label/*

Download and unpack the chicago dataset from [GoogleDrive](https://drive.google.com/file/d/1ZcZdUjGD8M0XOt7ssgMT6EXCNtAkLz7K/view?usp=sharing). Place the unpacked *chicago/* folder in *input/*. Your folder should look like this:
* input/
    * original/
        * image/
        * label/
    * chicago/
        * image/
        * label/

(Instead of downloading the processed images from drive you can also download the original *chicago.zip* from [zenodo](https://zenodo.org/record/1154821#.XyQB2nUzZhl). Process the images with *python TOMMY TODO*)

### Create Splits
Now create the dataset splits (train, validation) with *python 