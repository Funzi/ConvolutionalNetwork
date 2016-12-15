//
// Created by Filip Lux on 24.11.16.
//

#ifndef NN_NEURON_FCLAYER_H
#define NN_NEURON_FCLAYER_H

#include "Layer.h"

/**
*@brief FullConnected layer
*/
class FCLayer : public Layer{
public:

    double* ddot;
    double* out;
    double* w;

    double* input;
    double* down_ddot;


    /**
    *@brief constructor for the first layer
    *@param inputs number of inputs
    *@param neurons number of neurons
    */
    FCLayer(int &inputs, int &neurons);

    /**
    *@brief constructor for the upper layers
    *@param inputs number of inputs
    *@param neurons number of neurons
    *@param lower lower layer
    */
    FCLayer(int inputs, int neurons, Layer* lower);

    /**
    *@brief forward
    */
    void forward_layer();

    /**
    * @brief backpropagation
    */

    void backProp_layer();

    /**
    * @brief backpropagation for last layer
    * @param result expected values
    */
    void computeError(double* result);

    /**
    * @brief print weights
    */

    void learn();
    /**
    * @brief unsert new values to input
    */
    void update_input(double* in);

    /**
    * @brief print
    */
    void print();

	void setResults(double* results);

    /**
    * @brief destructor
    */
    ~FCLayer();


};


#endif //NN_NEURON_FCLAYER_H
