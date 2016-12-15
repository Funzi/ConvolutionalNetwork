//
// Created by Filip Lux on 24.11.16.
//

#include <iostream>
#include <vector>

#include "FCLayer.h"



// fullConnected layer
FCLayer::FCLayer(int inputs, int neurons, Layer* lower) {//creates layer, number of inputs and neurons


    n = neurons;
    down = lower;
    depth = 1;
    in = (down->n) * (down->depth);
	down_ddot = down->ddot;
    input = down->out;
    down->ou = n;
    ddot = new double[n];
    out = new double[n];
    w = new double[in*n];
    bias = new double[n];



    for (int i = 0; i < in*n; ++i) { //randomly initializes weights
        w[i] = fRand(INIT_MIN,INIT_MAX);
    }
    for (int i = 0; i < n; ++i) { //randomly initializes weights
        bias[i] = fRand(INIT_MIN,INIT_MAX);
    }
}

FCLayer::FCLayer(int inputs, int neurons) {
    n = neurons;
    in = inputs;
    ddot = new double[n];
    out = new double[n];
    w = new double[in*n];
    bias = new double[n];

    for (int i = 0; i < in*n; ++i) { //randomly initializes weights
        w[i] = fRand(INIT_MIN,INIT_MAX);
    }
    for (int i = 0; i < n; ++i) { //randomly initializes weights
        bias[i] = fRand(INIT_MIN,INIT_MAX);
    }

};

FCLayer::~FCLayer() {
    delete bias;
    delete ddot;
    delete out;
    delete w;
};

void FCLayer::forward_layer() { //step forward with activation function
    double sum = 0;
    double cs = -4;
    for (int i = 0; i < n; i++) {
        out[i] = bias[i];
        for (int j = 0; j < in; j++) {
            cs = out[i];
            out[i] += w[i*in+j] * input[j];

        }
        //out[i] = exp(out[i]);
        //sum += out[i];
        out[i] = sigma(out[i]);
        //sum = out[i];
        //sum += bsum;
    }
    //for (int i = 0; i < n; i++) {
    //    out[i] = out[i] / sum;
    //}
}

void FCLayer::backProp_layer() {
    //derivate of activation function

    for (int i = 0; i < n; i++) {
        ddot[i] *= out[i] * (1-out[i]);
    }

    if (down != NULL)
        for (int i = 0; i < in; i++) {
            down_ddot[i] = 0;
            for (int j = 0; j < n; j++) {
               down_ddot[i] += ddot[j] * w[i+j*n];
            }
        }
}

void FCLayer::updateDDot(double* error) {
    for (int i = 0; i < n; i++) {
        //ddot[i] = (out[i] - result[i]) * out[i] * (1-out[i]);
        ddot[i] = error[i];
    }
}

void FCLayer::print() {
    std::cout << "Results:" << std::endl;
	std::vector<double> sortedResults;
    for (int i = 0; i < n; i++) {
        std::cout << i+1 << ": " << out[i] << " ||| ";
    }
    std::cout << std::endl;
}

void FCLayer::learn() {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j<in; j++) {
            w[i*in+j] -= ddot[i] * input[j] * LR;
        }
    }
}

void FCLayer::update_input(double* in) {
    input = in;
};


