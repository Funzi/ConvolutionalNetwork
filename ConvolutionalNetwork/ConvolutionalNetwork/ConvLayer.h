//
// Created by bobby on 27.11.16.
//

#include <vector>
#include <sstream>
#include <iostream>
#include "Layer.h"

#ifndef NN_NEURON_CONVLAYER_H
#define NN_NEURON_CONVLAYER_H



class ConvLayer : public Layer {

public:

    /**
    *@brief number of neurons in one layer of filter
    */
    int wn;
    /**
    *@brief dimension of whole net
    */
    int depth, w_dim, s;
    /**
    *@brief deep of lower layer
    */
    int input_depth;

    ConvLayer(int filter_dim, int stroke, int filters, int in_dim, int in_depth, double* input);

    ConvLayer(int filter_dim, int stroke, int filters, Layer* lower);

    ConvLayer(std::string info, double* input);

    ConvLayer(std::string info, Layer* lower);


    void forward_layer();

    void backProp_layer();

    void learn();

    void update_input(double* in);

    ~ConvLayer();


    void print();

    std::string printLayer();

    void loadLayer(std::string line, int &filter_dim, int &stroke, int &filters, int &in_dim, int &in_depth, std::vector<double> &weights);
};


#endif //NN_NEURON_CONVLAYER_H
