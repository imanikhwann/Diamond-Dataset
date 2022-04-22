# Diamond Price Prediction Using Feedforward Neural Network
## !. Summary
The purpose of this project is to predict the price of a diamond based on the following features:
- 4C (cut, clarity, color, carat)
- dimensions (x, y, z)
- depth
- table
The dataset is obtained from https://www.kaggle.com/datasets/shivam2503/diamonds 
## 2. IDE and Framework
This project mainly used Spyder as IDE. The main framework used in this project are Pandas, Numpy, scikit-learn and TensorFlow Keras.
## 3. Methodology
### 3.1 Data Pipeline
The data are preprocessed using scikit-learn ordinal encoding on the categorical features. The unwanted column of the data are removed. The data are split into 70:15:15 training, validation and test ratio.
### 3.2 Model Pipeline
A feedforward neural network is constructed that is catered specifically for regression problem. The structure of the model is shown in the figure below.

![model architecture](https://user-images.githubusercontent.com/92588131/164628446-6daa049a-5d99-4d9a-9783-b1b910f2c16f.png)

The model is trained with batch size of 64 and 30 epochs with early stoping to prevent overfitting issue. The training stopped after epoch 5 with MAE 1263 and validation MAE 649.
## 4. Results
Prediction are made on the test data. The evaluation of the test data prediction are shown in the figure below.

![test_prediction](https://user-images.githubusercontent.com/92588131/164628429-174e4c85-1b6f-438b-9096-a82671a4c109.PNG)

Figure below shows the graph of predictions versus labels of the test data. 

![result](https://user-images.githubusercontent.com/92588131/164628387-900bbc05-c2e9-482e-a343-9cc2566c1f0e.png)
