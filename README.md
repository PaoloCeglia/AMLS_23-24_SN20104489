# AMLS_23-24_SN20104489

# Project Structure:

The project contains 4 executable commands accessible from the main.py

For task A:
* 'trainCNNA' : Trains a Convoluted Neural Network using the PneumoniaMNIST dataset and tests it.
* 'test_trained_model_B' : Tests a pretrained CNN model.

For task B:
* 'trainCNNB' : Trains a Convoluted Neural Network using the PneumoniaMNIST dataset and tests it.
* 'test_trained_model_B' : Tests a pretrained CNN model.

To execute those commands, uncomment one of the four functions and run 'main.py'

Please upload the 'pneumoniamnist.npz' and the 'pathmnist.npz' files inside the 'Datasets' directory.

# Role of each file:

'main.py' : Used to call one of the function described above.

Directory 'A':
* 'CNN_A.py' : Contains classes and methods for building, training, testing and evaluating the model for task A. 
* 'pretrained_CNNA.h5' : Is a pretrained model for task A.

Directory 'B':
* 'CNN_B.py' : Contains classes and methods for building, training, testing and evaluating the model for task B. 
* 'pretrained_CNNB.h5' : Is a pretrained model for task B.

The 'Datasets' directory must contain 'pneumoniamnist.npz' and 'pathmnist.npz' for the code to work properly.

# Libraries Used:

* tensorflow
* numpy
* sklearn
* matplotlib
* seaborn
