#ifndef _NNET_H_
#define _NNET_H_

#include <math.h>
#include <Eigen/Dense>



 
class NeuralNetwork{
    public:
        NeuralNetwork(const int input_nodes, const int hidden_nodes, const int output_nodes, 
                const float learning_rate, Eigen::Matrix3f whi, Eigen::Matrix3f who);
        void train(Eigen::Vector3f input_list, Eigen::Vector3f targets_list);
        Eigen::MatrixXf query(Eigen::Vector3f inputs);

    private:
        int input_nodes;
        int hidden_nodes;
        int output_nodes;
        float learning_rate;

        Eigen::Matrix3f whi;
        Eigen::Matrix3f who;

};


#endif
