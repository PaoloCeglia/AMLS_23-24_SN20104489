from A.CNN_A import trainCNNA, test_trained_model_A
from B.CNN_B import trainCNNB, test_trained_model_B

#Loads the data sets and pretrained files
data_path_A = 'Datasets/pneumoniamnist.npz'
model_path_A = 'A/pretrained_CNNA.h5'
data_path_B = 'Datasets/pathmnist.npz'
model_path_B = 'B/pretrained_CNNB.h5'

def main():

#  Uncomment one of the following lines based on the action required:

#### TASK A ####


    #trainCNNA(data_path_A)                            # Train and evaluate the CNN for Task A

    #test_trained_model_A(model_path_A, data_path_A)   # Test with pretrained model A


#### TASK B ####


    #trainCNNB(data_path_B)                             # Train and evaluate the CNN for Task B

    #test_trained_model_B (model_path_B, data_path_B)   # Test with pretrained model B
    
if __name__ == '__main__':
    main()
