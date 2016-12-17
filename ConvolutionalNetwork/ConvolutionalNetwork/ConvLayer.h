//
// Created by Filip Lux on 27.11.16.
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
    int w_dim, s;
    /**
    *@brief deep of lower layer
    */

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


    /**
     * @brief prints info about layer
     * @return string describing layer
     */
    std::string printLayer();

    /**
     * @brief creates layer from the file
     * @param line
     * @param filter_dim
     * @param stroke
     * @param filters
     * @param in_dim
     * @param in_depth
     * @param weights
     */
    void loadLayer(std::string line, int &filter_dim, int &stroke, int &filters, int &in_dim, int &in_depth, std::vector<double> &weights, std::vector<double> &bias);
};


#endif //NN_NEURON_CONVLAYER_H