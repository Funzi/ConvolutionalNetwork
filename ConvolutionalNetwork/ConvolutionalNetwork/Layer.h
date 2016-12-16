//
// Created by bobby on 2.12.16.
//

#ifndef FULLCONNECTEDNEURALNET_LAYER_H
#define FULLCONNECTEDNEURALNET_LAYER_H

#include <cmath>        //include exp
#include <algorithm>    //include rand
#define INIT_MAX 0.1  //max initialize weight
#define INIT_MIN -0.1 //min initialize weight
#define LR 0.001  //learning rate


//random function used for initialisation of weights
static double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}


//activation function
static double sigma(double x) {
    return 1/(1+exp(-x));
}

class Layer {
public:
    /**
    *@brief number of neurons in this layer, in lower layer, in upper layer
    */
    int n, in, ou;
    /**
    *@brief pointer to upper layer, NULL if layer is output
    */
    Layer* up;
    /**
    *@brief pointer to lower layer, NULL if layer is input
    */
    Layer* down;
    /**
    *@brief output of neurons
    */
    double* out;
    /**
    *@brief ddot of neurons
    */
    double* ddot;
    /**
    *@brief weights of neurons
    */
    double* w;
    /**
    *@brief bias of layer
    */
    double* bias;

    double* input;
    double* down_ddot;
    int depth;
    int input_dim;
    int dim;

    /**
    * @brief backpropagation
    */
    virtual void backProp_layer() = 0;

    /**
    * @brief forward pass
    */
    virtual void forward_layer() = 0;
    /**
    * @brief changing weights according ddot
    */
    virtual void learn() = 0;
    /**
    * @brief insert new values to input
    */
    virtual void update_input(double* in) = 0;

    /**
    * @brief print
    */
    virtual void print() = 0;

    /**
    * @brief printLayer
    */
    virtual std::string printLayer()=0;



};

#endif //FULLCONNECTEDNEURALNET_LAYER_H
