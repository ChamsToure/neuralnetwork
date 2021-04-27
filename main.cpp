#include <iostream>
#include "nnet.h"

using namespace std;

int main() {
    const int inputnodes = 3;
    const int hiddennodes = 3;
    const int outputnodes = 3;
    
    Eigen::Matrix3f w_input_hidden;
    Eigen::Matrix3f w_output_hidden;
    w_input_hidden << 0.9, 0.3, 0.4, 0.2, 0.8, 0.2, 0.1, 0.5, 0.6;
    w_output_hidden << 0.3, 0.7, 0.5, 0.6, 0.5, 0.2, 0.8, 0.1, 0.9; 

    const float learning_rate = 0.3;
    
    NeuralNetwork neuralnet = NeuralNetwork(inputnodes,hiddennodes,outputnodes,learning_rate, w_input_hidden, w_output_hidden);
    Eigen::Vector3f inputs(1.0, 3.7, 4.1);
    cout << neuralnet.query(inputs);
    Eigen::Vector3f targets(0.8, 0.6, 0.8);
    neuralnet.train(inputs, targets);
}
