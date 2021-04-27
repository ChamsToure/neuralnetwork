#include "nnet.h"
#include <Eigen/Dense>
#include <iostream>

//We use sigmoid as the activation function in this case
template<typename T>
T sigmoid(T &x){
    T result = 1 / (1 + exp(-x));
    return result;
}

//Constructor
NeuralNetwork::NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes, float learning_rate, Eigen::Matrix3f whi, Eigen::Matrix3f who)
    : input_nodes{input_nodes}, hidden_nodes{hidden_nodes}, output_nodes{output_nodes}, learning_rate{learning_rate}, whi{whi}, who{who} {
        std::cout << "Neural Network constructed!\n"; 
};


Eigen::MatrixXf NeuralNetwork::query(Eigen::Vector3f inputs){
    //Transpose the rowvector
    inputs = inputs.transpose();
    //Calculate the hidden inputs
    Eigen::MatrixXf hidden_inputs = whi * inputs;
    //Copy the hidden inputs into hidden outputs to execute the activation function with each value
    
    Eigen::MatrixXf hidden_outputs = hidden_inputs;
    for (int i = 0; i < hidden_outputs.rows(); i++)
            hidden_outputs(i,0) = sigmoid(hidden_outputs(i,0));

    //Calculate the final outputs of the neural network
    Eigen::MatrixXf final_inputs = who * hidden_outputs;
    Eigen::MatrixXf final_outputs = final_inputs;

    //Apply the sigmoid function to the output layer
    for (int i = 0; i < final_outputs.rows(); i++)
            final_outputs(i,0) = sigmoid(final_outputs(i,0));

    return final_outputs;
};

//Backpropagation
//targets_list is a list with the correct values
void NeuralNetwork::train(Eigen::Vector3f input_list, Eigen::Vector3f targets_list) {
    //Transpose both rowvectors
    input_list = input_list.transpose();
    targets_list = targets_list.transpose();

    //Calculate the hidden inputs
    Eigen::MatrixXf hidden_inputs = whi * input_list;
    //Copy the hidden inputs into hidden outputs to execute the activation function with each value
    
    Eigen::MatrixXf hidden_outputs = hidden_inputs;
    for (int i = 0; i < hidden_outputs.rows(); i++)
            hidden_outputs(i,0) = sigmoid(hidden_outputs(i,0));

    //Calculate the final outputs of the neural network
    Eigen::MatrixXf final_inputs = who * hidden_outputs;
    Eigen::MatrixXf final_outputs = final_inputs;

    //Apply the sigmoid function to the output layer
    for (int i = 0; i < final_outputs.rows(); i++)
            final_outputs(i,0) = sigmoid(final_outputs(i,0));    

    //output layer error
    Eigen::MatrixXf output_errors = targets_list - final_outputs;
    /*To calculate the hidden layer error, we need to split the output_errors
     *by the weights, and recombine it with the hidden nodes*/
    Eigen::MatrixXf hidden_errors = who.transpose() * output_errors;

    //Creating 3 by 1 Matrix to subtract each output value from 1
    Eigen::Matrix<float, 3, 1> ones;
    ones << 1,1,1;

    //Update the weights between hidden and output layers
    who += learning_rate * (output_errors * final_outputs.cwiseAbs() * (ones - final_outputs.cwiseAbs())) * (hidden_outputs.transpose());

    //Update the weights between the input and hidden layers
    whi += learning_rate * (hidden_errors * hidden_outputs.cwiseAbs() * (ones - hidden_outputs)) * (input_list.transpose());
}
