//
// Created by bobby on 27.11.16.
//

#ifndef NN_NEURON_POOLLAYER_H
#define NN_NEURON_POOLLAYER_H


#include <vector>
#include <sstream>
#include "Layer.h"

class PoolLayer : public Layer{

public:
    /**
     * Constructor
     * @param lower pointer to lower layer
     */
    PoolLayer(Layer* lower);

    /**
     * Destructor
     * @param lower pointer to lower layer
     */
    ~PoolLayer();


    /**
     * @brief forward pass
     */
    void forward_layer();

    /**
    * @brief backpropagation
    */
    void backProp_layer();

    /**
    * @brief changing weights according ddot
    */
    void learn();
    /**
    * @brief insert new values to input
    */
    void update_input(double* in);

    /**
    * @brief print
    */
    void print();

    std::string printLayer();

};


#endif //NN_NEURON_POOLLAYER_H
