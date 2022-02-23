# Dog Breed Classification Using Transfer Learning
#### Matthew Zirpoli

## Abstract
The goal of this project was to create a deep learning model to classify dog breeds. I worked with a dataset from the Kaggle Dog Breed Identification competition, which provided an ImageNet subset of only dogs. The data contained a training set of over 10,000 images with ground truth labels, as well as an equal amount of images for testing the final model. Transfer learning was used to apply MobileNetV2, a fast performing and accurate CNN with proven results on ImageNet samples, to this problem domain. Images were pre-processed using batching to optimize model performance. Data augmentation techniques were applied to mitigate overfitting to the training set. A final deep learning model produced 90% accuracy on the test set.

## Design
A fast growing trend in the pet industry is genetic testing of your furry friend to determine its breed. The results of this testing are useful because they inform important ownership decisions like the approach to training, and which health risks to be on the lookout for. A barrier to entry for these tests is that they are too expensive for many owners, and may not be within budget for foster or volunteer organizations who have many pets in and out of their care. A potential alternative for these expensive tests is an image classification model, which can predict the breed of a dog from only a picture!

## Data
The dataset used for this project was the [Dog Breed Identification] (https://www.kaggle.com/c/dog-breed-identification/overview) community competition on Kaggle. The set consisted of two sets of images, train and test, along with a .csv file containing file and label pairs for the training set. The training set contained 10,222 images of dogs of 120 different breeds. Each breed had between 66 and 126 images of the breed in the training set, which was sufficient data to train on, as well as fairly evenly distributed. The test set contained 10,357 images to make predictions for. Additional data was sourced from friends and family, for testing images of their dogs!

## Algorithms
### Preprocessing Data
Ground truth labels were processed into one-hot encoded arrays to represent the true label of each image.

Batch processing of images was employed to optimize performance on CPU.Images were processed into tensors with 3 color channels, re-sized to 224x224px to match the required model input, and re-scaled to 0-1. This was performed on train, validation, and test sets.

### Transfer Learning Model Selection
[MobileNetV2] (https://arxiv.org/pdf/1801.04381v4.pdf) was selected for transfer learning. This model was chosen due to its performance, and relation to the problem domain. This model was designed to be distributed through mobile, leading to the belief that it would perform well even on my outdated system, allowing for several training runs to enable different approaches within the constrained time. Additionally, this model was tested against ImageNet classification benchmarks, which the dataset is sampled from.

### Data Augmentation
To mitigate issues with overfitting on the training set, training images were augmented randomly prior to input. Augmentations were applied using RNG and seed splitting to randomly apply crop, brightness adjustment, and horizontal flipping. Vertical flipping, rotation, and contrast were used as well, but dropped from the final model's data augmentation pre-processing steps as model accuracy began dropping with too many operations applied.

### Evaluation and Selection
To evaluate each approach, training data was split 80/20 into train and validation sets. Training sets were augmented, while validation sets weren't. Accuracy was used as the primary performance metric to determine perfomance. Iteration through several augmentation approaches arrived at a final model, which was applied to the test set. A final accuracy of 0.90482 was achieved on the test set. 

## Tools
- Numpy and Pandas for data manipulation
- Matplotlib and Seaborn for data visualization
- TensorFlow/Keras for image processing, augmentation and deep learning algorithms
- TensorFlow Hub for transfer learning model
  - [MobileNetV2] (https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5)

## Communication 
The findings and slide deck accompanying this project's presentation are accessible via this GitHub repo.
