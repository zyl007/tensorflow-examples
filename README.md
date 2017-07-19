# `Images/` : Stanford Dogs 数据集


Stanford Dogs Dataset 

http://vision.stanford.edu/aditya86/ImageNetDogs/

Aditya Khosla     Nityananda Jayadevaprakash     Bangpeng Yao     Li Fei-Fei

Stanford University

The Stanford Dogs dataset contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. Contents of this dataset:

    Number of categories: 120
    Number of images: 20,580
    Annotations: Class labels, Bounding boxes
    
## README.md for Stanford Dogs Dataset**

```
Stanford Dogs Dataset
---------------------

For more information about the dataset, please visit the dataset website:
  http://vision.stanford.edu/aditya86/ImageNetDogs/

If you use this dataset in a publication, please cite the dataset as
described on the website.

File Information:
 - images/
    -- Images of different breeds are in separate folders
 - annotations/
    -- Bounding box annotations of images
 - file_list.mat
    -- List of all files in the dataset
 - train_list.mat
    -- List and labels of all training images in dataset
 - test_list.mat
    -- List and labels of all test images in dataset

Train splits:
 In order to test with fewer than 100 images per class, the first
 n indices for each class in train_list.mat were used, where n is
 the number of training images per class.

Features (train_data.mat, test_data.mat):
 - train_data/test_data
   -- contains the feature matrix after histogram intersection kernel has been applied
 - train_fg_data/test_fg_data
   -- contains the feature matrix before applying the histogram intersection kernel
 - train_info/test_info
   -- contains the labels and ids for the corresponding image in the feature matrix

For more information, please contact Aditya Khosla at aditya86@cs.stanford.edu.
```