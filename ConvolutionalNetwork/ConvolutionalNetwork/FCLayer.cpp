//
// Created by Filip Lux on 24.11.16.
//

#include <iostream>
#include <vector>

#include "FCLayer.h"



// fullConnected layer
FCLayer::FCLayer(int inputs, int neurons, Layer* lower) { //creates layer, number of inputs and neurons
    n = neurons;
    in = inputs;
    down = lower;
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

FCLayer::FCLayer(int &inputs, int &neurons) {
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
    for (int i = 0; i < n; i++) {
        out[i] = bias[i];
        for (int j = 0; j < in; j++) {
            out[i] += w[i*in+j] * input[j];
        }
        out[i] = sigma(out[i]);
    }
}

void FCLayer::backProp_layer() {
    for (int i = 0; i < n; i++) {
        down_ddot[i] = 0;
        for (int j = 0; j < ou; j++) {
           down_ddot[i] += ddot[j] * w[i+j*ou];
        }
    }
}

void FCLayer::computeError(double* result) {
    for (int i = 0; i < n; i++) {
        ddot[i] = (out[i] - result[i]) * out[i] * (1-out[i]);
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


std::string FCLayer::printLayer(){

    std::stringstream ss;

    ss << "type:ConvLayer|";
    ss << "inputs"<< in << "|";
    ss << "neurons"<< n << "|";
    ss << "weights:";

    long n_weights = in*n;
    for (int i = 0; i < n_weights; i++)
        ss << w[i] << ",";

    ss << "\n";

    std::string out = ss.str();
    ss.clear();

    return out;
};

void FCLayer::setResults(double* results) {
	for (int i = 0; i < 10; i++) {
		results[i] = out[i];
	}
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


