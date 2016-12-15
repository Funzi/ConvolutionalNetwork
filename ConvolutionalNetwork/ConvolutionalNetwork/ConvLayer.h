//
// Created by Filip Lux on 27.11.16.
//

#include <vector>
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
    int w_dim,s;
    /**
    *@brief deep of lower layer
    */


    ConvLayer(int filter_dim, int stroke, int filters, int in_dim, int in_depth, double* input);

    ConvLayer(int filter_dim, int stroke, int filters, Layer* lower);

    void forward_layer();

    void backProp_layer();

    void learn();

    void update_input(double* in);

    ~ConvLayer();


    void print();

};


#endif //NN_NEURON_CONVLAYER_H
