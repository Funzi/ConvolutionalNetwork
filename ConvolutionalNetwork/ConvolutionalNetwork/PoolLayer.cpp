//
// Created by Filip Lux on 27.11.16.
//

#include "PoolLayer.h"

PoolLayer::PoolLayer(Layer* lower) {

    down = lower;
    input = down->out;
    input_dim = down->dim;
    dim = input_dim/2;
    n = dim*dim;
    in = n*4;
    depth = down->depth;
    down_ddot = down->ddot;

    out = new double[n*depth];
    ddot = new double[n*depth];

}

void PoolLayer::forward_layer() {

    double sum;
    int b,s;
    for (int k = 0; k < depth; k++) {
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                b = k*in + 2 * i * input_dim + 2 * j;
                s = k*n + i * dim + j;

                sum = std::max(std::max(input[b], input[b + 1]),
                                      std::max(input[b + input_dim], input[b + input_dim + 1]));

                out[s] = sum;
                down_ddot[b] = (out[s] == input[b]) ? 1 : 0;
                down_ddot[b + 1] = (out[s] == input[b + 1]) ? 1 : 0;
                down_ddot[b + input_dim] = (out[s] == input[b + input_dim]) ? 1 : 0;
                down_ddot[b + input_dim + 1] = (out[s] == input[b + input_dim + 1]) ? 1 : 0;
            }
        }
    }

}

void PoolLayer::backProp_layer() {

    int d;
    for (int k = 0; k < depth; k++) {
        for (int i = 0; i < input_dim; i++) {
            for (int j = 0; j < input_dim; j++) {
                d = i * input_dim + j;

                down_ddot[k * in + d] *=  ddot[k * n + i/2 * dim + j/2];
            }
        }
    }
}

void PoolLayer::learn() {
    //does nothing
}

void PoolLayer::update_input(double* in) {
    //not implemented
}

void PoolLayer::print() {
    //not implemented
}


PoolLayer::~PoolLayer() {
    delete []ddot;
    delete []out;
}