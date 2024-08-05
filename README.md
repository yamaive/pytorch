# CAPSTONE Project

The goal of this project is to build and train a deep learning model for image classification using PyTorch. Using the dataset, I will implement a deep neural network and evaluate its performance.

## About Dataset

The dataset is Kaggle's FER2013 dataset. This dataset consists of 48x48 pixel grayscale images and contains automatically registered face images. Each image has a face placed in the center, occupying approximately the same space within the image. Based on facial expressions, the images are classified into seven emotion categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The training set contains 28,709 examples and the public test set contains 3,589 examples.  
[FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013/data)

## Selected Deep Learning Architecture and Reasons for Selection

The architecture selected is Convolutional Neural Network (CNN).

- **Effective for Image Data:** CNN is very effective for image data, extracting spatial features. Convolutional layers can easily capture local features of an image (edges, textures, patterns, etc.), thereby efficiently learning the spatial structure of the image. This is very important for image classification tasks.
- **Parameter Efficiency:** CNNs share filters, which greatly reduces the number of parameters and improves computational efficiency. This speeds up model training and reduces memory usage.
- **Positional Invariance:** CNNs learn positional invariance of features by using pooling layers. This means the ability to correctly recognize objects in an image even if their position changes.
- **Versatility:** CNN can use pre-trained models (e.g., VGG16, ResNet, etc.) to perform transfer learning.
- **Proven Effectiveness:** CNN has proven its effectiveness by excelling in numerous image classification competitions.

## Define the Architecture of the Model and the Reasons

### Input Layer

- **Input size:** (48, 48, 1) - Size of grayscale images  
  This matches the image size of the dataset and is necessary to input the images properly into the model.

### Convolutional Layers

- **Conv2D:** Filters = 32, Kernel size = (3, 3), Activation function = ReLU
- **Conv2D:** Filters = 64, Kernel size = (3, 3), Activation function = ReLU

The first convolutional layer extracts low-level features (such as edges and textures) from the images. The subsequent convolutional layers increase the number of filters, allowing the model to learn more features.  
Starting with 32 filters instead of 64 in the first layer helps in keeping computational costs low while efficiently learning the low-level features of the image.  
The reason for the kernel size of (3,3) is that it is widely used in deep learning and has proven effective in image classification tasks. It also has high computational efficiency and fewer parameters, making the model training faster and more memory efficient.  
The ReLU activation function introduces non-linearity and mitigates the vanishing gradient problem. Specifically, ReLU fixes negative inputs to 0 and passes positive inputs as they are, enabling learning to progress in deep networks without gradient vanishing.

### Pooling Layer

- **MaxPooling2D:** Pool size = (2, 2)

MaxPooling layers reduce the size of the feature maps, lowering computational costs while retaining important features. Specifically, MaxPooling takes the maximum value in each pooling window, preserving the most prominent features. This helps prevent overfitting and improves positional invariance (the ability to recognize features regardless of where they appear in the image).  
Another option is AveragePooling, which takes the average value in each window, but it may blur prominent features compared to MaxPooling.

### Repeated Convolutional and Pooling Layers

- **Conv2D:** Filters = 128 or 64, Kernel size = (3, 3), Activation function = ReLU
- **MaxPooling2D:** Pool size = (2, 2)

Repeating the convolutional and pooling layers allows for the extraction of more advanced features and the learning of complex patterns within the image. This enables the model to make more accurate classifications. By using a repeated structure, the model learns different levels of abstraction, starting with low-level features (edges and textures) and progressing to high-level features (shapes and objects).  
Increasing the number of filters to 128 allows the model to generate more feature maps, capturing more detailed and complex patterns and higher-level features in the images. Using 64 filters maintains higher computational efficiency and lower memory usage.

### Flattening Layer

This step converts the 2D feature maps learned by the convolutional layers into a 1D vector, necessary for input into the fully connected layers. This allows the fully connected layers to perform the final classification based on these features.

### Fully Connected Layers

- **Dense:** Units = 128 or 64, Activation function = ReLU
- **Dropout:** Rate = 0.5 or 0.2 (Regularization technique)

The fully connected layers are used to perform the final classification based on the features extracted by the convolutional layers. Using 128 units provides sufficient learning capacity, and the ReLU activation function ensures non-linearity in the model.  
Dropout is used to prevent overfitting by randomly disabling units during training, preventing the model from becoming too reliant on specific units, and improving generalization ability. A dropout rate of 0.5 means that 50% of the units are disabled during training, a commonly used value. The dropout rate is often adjusted between 0.2 and 0.5, with 0.5 widely used in deep learning models.

### Output Layer

- **Dense:** Units = 7 (7 emotion categories), Activation function = Softmax

The output layer classifies each image into one of the 7 emotion categories. The Softmax activation function outputs the probability for each class, ensuring that the sum of probabilities is 1. This allows for easy interpretation of which emotion corresponds to each image.

This structure excels at feature extraction from image data and, by combining appropriate activation functions and regularization techniques, achieves a highly accurate emotion classification model.


## Model Architecture Diagram

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 48, 48]             320
              ReLU-2           [-1, 32, 48, 48]               0
            Conv2d-3           [-1, 64, 48, 48]          18,496
              ReLU-4           [-1, 64, 48, 48]               0
         MaxPool2d-5           [-1, 64, 24, 24]               0
            Conv2d-6          [-1, 128, 24, 24]          73,856
              ReLU-7          [-1, 128, 24, 24]               0
         MaxPool2d-8          [-1, 128, 12, 12]               0
            Linear-9                  [-1, 128]       2,359,424
             ReLU-10                  [-1, 128]               0
          Dropout-11                  [-1, 128]               0
           Linear-12                    [-1, 7]             903
          Softmax-13                    [-1, 7]               0
================================================================
Total params: 2,452,999
Trainable params: 2,452,999
Non-trainable params: 0
```

## Model Execution Results

```
Training Model1
Epoch 0/24, Loss: 1.8201
Epoch 1/24, Loss: 1.7376
Epoch 2/24, Loss: 1.6987
Epoch 3/24, Loss: 1.6717
Epoch 4/24, Loss: 1.6468
Epoch 5/24, Loss: 1.6279
Epoch 6/24, Loss: 1.6085
Epoch 7/24, Loss: 1.5759
Epoch 8/24, Loss: 1.5659
Epoch 9/24, Loss: 1.5612
Epoch 10/24, Loss: 1.5565
Epoch 11/24, Loss: 1.5481
Epoch 12/24, Loss: 1.5468
Epoch 13/24, Loss: 1.5417
Epoch 14/24, Loss: 1.5362
Epoch 15/24, Loss: 1.5385
Epoch 16/24, Loss: 1.5368
Epoch 17/24, Loss: 1.5355
Epoch 18/24, Loss: 1.5347
Epoch 19/24, Loss: 1.5337
Epoch 20/24, Loss: 1.5361
Epoch 21/24, Loss: 1.5320
Epoch 22/24, Loss: 1.5345
Epoch 23/24, Loss: 1.5333
Epoch 24/24, Loss: 1.5324
Evaluating Model1
Accuracy: 54.22%
```

## Post-Training Analysis

From the training results, the following points can be observed:

1. **Loss Reduction**: The initial loss starts at 1.8201 and decreases to 1.5324. The reduction in loss indicates that the model is learning.
2. **Accuracy**: The final accuracy is 54.22%. This is significantly higher than random guessing for seven classes (about 14%), but there is room for improvement.
3. **Loss Stabilization**: The decrease in loss becomes smaller as epochs progress, especially after epoch 15, where the loss plateaus and shows minimal improvement.

## Model Tuning

The Softmax function is used in the final layer to interpret the model's output as probabilities. When using CrossEntropyLoss as the loss function, the Softmax function is usually integrated within it, so there is no need to apply Softmax again within the loss function.

### Simplification of Gradient Calculation
Within CrossEntropyLoss, the calculation of Softmax and logarithms are combined, simplifying the gradient calculation. Specifically, it allows for a numerically stable method of calculating gradients for the Softmax output. This reduces the issues of vanishing or exploding gradients, leading to more stable training. Including Softmax in the model's output layer can cause redundant processing during loss calculation, making training unstable. Therefore, it is common to use raw scores as the model output and pass them directly to CrossEntropyLoss.

## Improved Model 2 Architecture Diagram

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 48, 48]             320
              ReLU-2           [-1, 32, 48, 48]               0
            Conv2d-3           [-1, 64, 48, 48]          18,496
              ReLU-4           [-1, 64, 48, 48]               0
         MaxPool2d-5           [-1, 64, 24, 24]               0
            Conv2d-6          [-1, 128, 24, 24]          73,856
              ReLU-7          [-1, 128, 24, 24]               0
         MaxPool2d-8          [-1, 128, 12, 12]               0
            Linear-9                  [-1, 128]       2,359,424
             ReLU-10                  [-1, 128]               0
          Dropout-11                  [-1, 128]               0
           Linear-12                    [-1, 7]             903
================================================================
Total params: 2,452,999
Trainable params: 2,452,999
Non-trainable params: 0
----------------------------------------------------------------
```

## Model 2 Execution Results

```
Training Model2
Epoch 0/24, Loss: 1.6653
Epoch 1/24, Loss: 1.4272
Epoch 2/24, Loss: 1.2928
Epoch 3/24, Loss: 1.1873
Epoch 4/24, Loss: 1.0908
Epoch 5/24, Loss: 1.0002
Epoch 6/24, Loss: 0.9054
Epoch 7/24, Loss: 0.7572
Epoch 8/24, Loss: 0.7124
Epoch 9/24, Loss: 0.6893
Epoch 10/24, Loss: 0.6659
Epoch 11/24, Loss: 0.6450
Epoch 12/24, Loss: 0.6258
Epoch 13/24, Loss: 0.6107
Epoch 14/24, Loss: 0.5888
Epoch 15/24, Loss: 0.5863
Epoch 16/24, Loss: 0.5786
Epoch 17/24, Loss: 0.5797
Epoch 18/24, Loss: 0.5753
Epoch 19/24, Loss: 0.5717
Epoch 20/24, Loss: 0.5696
Epoch 21/24, Loss: 0.5684
Epoch 22/24, Loss: 0.5707
Epoch 23/24, Loss: 0.5676
Epoch 24/24, Loss: 0.5709
Evaluating Model2
Accuracy: 57.72%
```

## Post-Training Analysis of Model 2

From the training results, the following points can be observed:

1. **Loss Reduction**: In Model 2, the loss significantly decreases from 1.6653 at epoch 0 to 0.5709 at epoch 24. The reduction in loss is significant, indicating good learning by the model.
2. **Accuracy**: The final accuracy of Model 2 is 57.72%, which is higher than Model 1's 54.22%.
3. **Loss Stabilization**: The loss in Model 2 becomes stable or slightly increases after epoch 15, indicating it might be a good time to stop training to avoid overfitting.

### Investigation of Optimal Hyperparameter Combinations

The batch size was fixed at 32, and the combinations of dropout rates (0.3, 0.5, 0.7) and learning rates (0.001, 0.01, 0.0001) were investigated.

#### Results of the Investigation:

**Training with batch_size=32, dropout_rate=0.3, learning_rate=0.001**
```
Epoch 0/9, Loss: 1.5866
Epoch 1/9, Loss: 1.3114
Epoch 2/9, Loss: 1.1660
Epoch 3/9, Loss: 1.0412
Epoch 4/9, Loss: 0.9040
Epoch 5/9, Loss: 0.7745
Epoch 6/9, Loss: 0.6651
Epoch 7/9, Loss: 0.4734
Epoch 8/9, Loss: 0.4238
Epoch 9/9, Loss: 0.3976
Accuracy: 57.30%
```

**Training with batch_size=32, dropout_rate=0.3, learning_rate=0.01**
```
Epoch 0/9, Loss: 1.7664
Epoch 1/9, Loss: 1.6276
Epoch 2/9, Loss: 1.6122
Epoch 3/9, Loss: 1.5959
Epoch 4/9, Loss: 1.5709
Epoch 5/9, Loss: 1.5577
Epoch 6/9, Loss: 1.5457
Epoch 7/9, Loss: 1.4722
Epoch 8/9, Loss: 1.4478
Epoch 9/9, Loss: 1.4394
Accuracy: 42.03%
```

**Training with batch_size=32, dropout_rate=0.3, learning_rate=0.0001**
```
Epoch 0/9, Loss: 1.6946
Epoch 1/9, Loss: 1.5248
Epoch 2/9, Loss: 1.4208
Epoch 3/9, Loss: 1.3522
Epoch 4/9, Loss: 1.2950
Epoch 5/9, Loss: 1.2460
Epoch 6/9, Loss: 1.1953
Epoch 7/9, Loss: 1.1231
Epoch 8/9, Loss: 1.1095
Epoch 9/9, Loss: 1.0985
Accuracy: 52.95%
```

**Training with batch_size=32, dropout_rate=0.5, learning_rate=0.001**
```
Epoch 0/9, Loss: 1.6432
Epoch 1/9, Loss: 1.3925
Epoch 2/9, Loss: 1.2718
Epoch 3/9, Loss: 1.1739
Epoch 4/9, Loss: 1.0738
Epoch 5/9, Loss: 0.9835
Epoch 6/9, Loss: 0.8954
Epoch 7/9, Loss: 0.7393
Epoch 8/9, Loss: 0.7011
Epoch 9/9, Loss: 0.6693
Accuracy: 55.98%
```

**Training with batch_size=32, dropout_rate=0.5, learning_rate=0.01**
```
Epoch 0/9, Loss: 1.8826
Epoch 1/9, Loss: 1.8120
Epoch 2/9, Loss: 1.8111
Epoch 3/9, Loss: 1.8109
Epoch 4/9, Loss: 1.8109
Epoch 5/9, Loss: 1.8106
Epoch 6/9, Loss: 1.8108
Epoch 7/9, Loss: 1.8100
Epoch 8/9, Loss: 1.8098
Epoch 9/9, Loss: 1.8098
Accuracy: 24.71%
```

**Training with batch_size=32, dropout_rate=0.5, learning_rate=0.0001**
```
Epoch 0/9, Loss: 1.7194
Epoch 1/9, Loss: 1.5756
Epoch 2/9, Loss: 1.4865
Epoch 3/9, Loss: 1.4190
Epoch 4/9, Loss: 1.3728
Epoch 5/9, Loss: 1.3267
Epoch 6/9, Loss: 1.2885
Epoch 7/9, Loss: 1.2223
Epoch 8/9, Loss: 1.2168
Epoch 9/9, Loss: 1.2096
Accuracy: 51.59%
```

**Training with batch_size=32, dropout_rate=0.7, learning_rate=0.001**
```
Epoch 0/9, Loss: 1.6846
Epoch 1/9, Loss: 1.4648
Epoch 2/9, Loss: 1.3668
Epoch 3/9, Loss: 1.2923
Epoch 4/9, Loss: 1.2367
Epoch 5/9, Loss: 1.1789
Epoch 6/9, Loss: 1.1340
Epoch 7/9, Loss: 1.0359
Epoch 8/9, Loss: 1.0079
Epoch 9/9, Loss: 0.9956
Accuracy: 56.26%
```

**Training with batch_size=32, dropout_rate=0.7, learning_rate=0.01**
```
Epoch 0/9, Loss: 1.8643
Epoch 1/9, Loss: 1.8110
Epoch 2/9, Loss: 1.8105
Epoch 3/9, Loss: 1.8109
Epoch 4/9, Loss: 1.8105
Epoch 5/9, Loss: 1.8109
Epoch 6/9, Loss: 1.8108
Epoch 7/9, Loss: 1.8104
Epoch 8/9, Loss: 1.8098
Epoch 9/9, Loss: 1.8098
Accuracy: 24.71%
```

**Training with batch_size=32, dropout_rate=0.7, learning_rate=0.0001**
```
Epoch 0/9, Loss: 1.7725
Epoch 1/9, Loss: 1.6345
Epoch 2/9, Loss: 1.5601
Epoch 3/9, Loss: 1.5067
Epoch 4/9, Loss: 1.4692
Epoch 5/9, Loss: 1.4327
Epoch 6/9, Loss: 1.4085
Epoch 7/9, Loss: 1.3661
Epoch 8/9, Loss: 1.3619
Epoch 9/9, Loss: 1.3490
Accuracy: 50.28%
```

#### Best Accuracy: 57.30%

#### Best Hyperparameters:
- **batch_size:** 32, **dropout_rate:** 0.3, **learning_rate:** 0.001

### Explanation of Dropout Rate and Learning Rate Settings
#### Dropout Rate
**Setting 1: 0.3**
- During training, 30% of the neurons in each layer are randomly deactivated.
- This helps prevent overfitting by ensuring the network does not become overly reliant on specific neurons.
  
#### Learning Rate
**Setting 1: 0.001**
- A higher learning rate results in larger updates to the model's weights.
- A value of 0.001 is commonly used as it allows relatively fast learning, but if it's too high, there is a risk of the model's loss function diverging.

### Reinvestigation of Settings
**Setting 1 (0.3 dropout rate, 0.001 learning rate)** leads to faster learning but comes with a higher risk of overfitting.
**Setting 2 (0.7 dropout rate, 0.0001 learning rate)** strongly prevents overfitting but may result in slower learning.

### Re-running with Epochs Set to 15:
```
Training with dropout_rate=0.3, learning_rate=0.001
Epoch 0/14, Loss: 1.5639, Accuracy: 38.62%
Epoch 1/14, Loss: 1.2878, Accuracy: 50.93%
Epoch 2/14, Loss: 1.1451, Accuracy: 56.56%
Epoch 3/14, Loss: 1.0041, Accuracy: 62.04%
Epoch 4/14, Loss: 0.8778, Accuracy: 66.91%
Epoch 5/14, Loss: 0.7404, Accuracy: 72.05%
Epoch 6/14, Loss: 0.6313, Accuracy: 75.78%
Epoch 7/14, Loss: 0.4544, Accuracy: 83.02%
Epoch 8/14, Loss: 0.4015, Accuracy: 84.70%
Epoch 9/14, Loss: 0.3740, Accuracy: 85.74%
Epoch 10/14, Loss: 0.3490, Accuracy: 86.60%
Epoch 11/14, Loss: 0.3262, Accuracy: 87.62%
Epoch 12/14, Loss: 0.3052, Accuracy: 88.14%
Epoch 13/14, Loss: 0.2913, Accuracy: 88.83%
Epoch 14/14, Loss: 0.2758, Accuracy: 89.47%
Dropout Rate: 0.3, Learning Rate: 0.001, Test Accuracy: 57.80%
```

<img alt="image" src="https://github.com/user-attachments/assets/09ea5bcb-2ef7-4171-830e-5debc9d37100">

### Confusion Matrix Analysis
The confusion matrix shows how accurately (or inaccurately) the model predicted each emotion category (class). The rows represent the actual labels (true emotions), and the columns represent the predicted labels (predicted emotions).

- **Happy**: Correctly predicted 1399 times, the highest among all emotions.
- **Fear**: Prone to misclassification with other emotions.
<img alt="image" src="https://github.com/user-attachments/assets/d31d0fd3-8690-437e-bda9-9bad1e72bb84">


### ROC Curve Analysis
The ROC curve (Receiver Operating Characteristic curve) illustrates the performance of the classification model for each emotion category. It plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

- **AUC 0.85 Classes**: Indicates the model performs very well. Emotions like Happy and Surprise are well identified by this model.
- **AUC 0.66 Class**: Indicates room for improvement. Fear may need more data or better diversity within the class.
<img alt="image" src="https://github.com/user-attachments/assets/c102882a-d682-44f0-bdd6-2275dae7eac3">

```
Classification Report:
              precision    recall  f1-score   support

       angry       0.47      0.50      0.48       958
     disgust       0.80      0.42      0.55       111
        fear       0.43      0.40      0.42      1024
       happy       0.76      0.79      0.77      1774
     neutral       0.51      0.53      0.52      1233
         sad       0.45      0.45      0.45      1247
    surprise       0.76      0.72      0.74       831

    accuracy                           0.58      7178
   macro avg       0.60      0.54      0.56      7178
weighted avg       0.58      0.58      0.58      7178
```
### Class-wise Performance Analysis
- **Angry**
  - **Accuracy**: 47%
  - **Recall**: 50%
  - **F1 Score**: 48%
  - Shows average performance.

- **Disgust**
  - **Accuracy**: 80%
  - **Recall**: 42%
  - **F1 Score**: 55%
  - High accuracy but low recall, indicating many Disgust cases are missed.

- **Fear**
  - **Accuracy**: 43%
  - **Recall**: 40%
  - **F1 Score**: 42%
  - Low performance overall.

- **Happy**
  - **Accuracy**: 76%
  - **Recall**: 79%
  - **F1 Score**: 77%
  - Indicates high performance across all metrics.

- **Neutral**
  - **Accuracy**: 51%
  - **Recall**: 53%
  - **F1 Score**: 52%
  - Average performance.

- **Sad**
  - **Accuracy**: 45%
  - **Recall**: 45%
  - **F1 Score**: 45%
  - Shows average performance.

- **Surprise**
  - **Accuracy**: 76%
  - **Recall**: 72%
  - **F1 Score**: 74%
  - Indicates relatively high performance.


### Reasons for High Accuracy in Predicting "Happy"

- **Clarity of Features**: Smiling involves very distinctive facial changes compared to other emotions. Features like the upward curve of the mouth and narrowed eyes make it easily distinguishable from other emotions.

- **Data Balance**: Many datasets, including FER-2013, often contain more images of smiles compared to other emotions. This abundance of examples allows the model to learn better and perform better in recognizing smiles.

- **Intensity of Expression**: Smiles are a relatively strong and clear emotional expression, making even slight changes easily recognizable. This makes classification easier compared to subtler emotions like "Neutral" or "Sad".

- **Quality of Data**: Images of smiles are often of higher quality. When capturing positive emotions, lighting and composition are usually better, which contributes to the ease of recognition.

## Project Summary

Building and Training a Deep Learning Model
I learned how to build and train a deep learning model for image classification using PyTorch. I selected a Convolutional Neural Network (CNN) architecture and understood its advantages, such as suitability for image data, parameter efficiency, position invariance, versatility, and proven effectiveness. I defined the model's architecture, designing the input layer, convolutional layers, pooling layers, fully connected layers, and output layer. By combining appropriate activation functions and normalization techniques, I developed a sentiment classification model.

Model Improvement
In the initial run, the model's accuracy was 54.22%. When using CrossEntropyLoss as the loss function, the Softmax function is embedded internally, so it is unnecessary to include it in the model's output layer. Including the Softmax function in the output layer can lead to redundant processing during loss calculation, potentially destabilizing training. This change improved the accuracy of Model 2 to 57.72%.

Hyperparameter Tuning
I fixed the batch size at 32 and explored combinations of dropout rates (0.3, 0.5, 0.7) and learning rates (0.001, 0.01, 0.0001). The combination of a dropout rate of 0.3 and a learning rate of 0.001 showed the best performance with a test accuracy of 57.80%.

Challenges and Future Improvements
While the model achieved a satisfactory accuracy of 57.80% for sentiment classification, there is room for improvement. The batch size, which refers to the number of data samples used in one iteration of training, can impact the stability and speed of convergence. Choosing an appropriate batch size can enhance the model's generalization performance. Additionally, optimizers are algorithms used to update model parameters, and different types, such as SGD, Adam, and RMSprop, have unique characteristics. Selecting the optimal optimizer can increase learning efficiency and achieve higher accuracy. By properly tuning these hyperparameters, it is expected to enhance the model's expressiveness, prevent overfitting, and ultimately achieve higher accuracy.

## Project Reflections

Through this project, I gained a deep understanding of the importance of CNN design and optimization, particularly the impact of hyperparameter tuning on model performance. I also learned the significance of each step in data preprocessing and model construction for image classification tasks.

I understood that image classification tasks require significant computing power. Image data is usually much larger than text data, and image classification involves processing information from each pixel, increasing the computational complexity. Furthermore, image classification often uses complex models like Convolutional Neural Networks (CNNs). These models have many parameters and require considerable computational resources for training.




