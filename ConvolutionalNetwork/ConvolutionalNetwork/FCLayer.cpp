//
// Created by Filip Lux on 24.11.16.
//

#include <iostream>
#include <vector>

#include "FCLayer.h"



// fullConnected layer
FCLayer::FCLayer(int inputs, int neurons, Layer* lower) { //creates layer, number of inputs and neurons

    n = neurons;
    down = lower;
    depth = 1;
    in = inputs;
    //in = (down->n) * (down->depth);
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
    depth = 1;
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

FCLayer::FCLayer(std::string layerInfo, Layer* lower) { //creates layer, number of inputs and neurons

    int neurons, inputs;
    std::vector<double> weights, vbias;
    loadLayer(layerInfo, neurons, inputs, weights, vbias);

    n = neurons;
    in = inputs;
    down = lower;
    depth = 1;
    down_ddot = down->ddot;
    input = down->out;
    down->ou = n;
    ddot = new double[n];
    out = new double[n];
    w = new double[in*n];
    bias = new double[n];

    for (int i = 0; i < in*n; ++i) { //randomly initializes weights
        w[i] = weights[i];
    }
    for (int i = 0; i < n; ++i) { //randomly initializes weights
        bias[i] = vbias[i];
    }
}

FCLayer::FCLayer(std::string layerInfo) {

    int neurons, inputs;
    std::vector<double> weights, vbias;
    loadLayer(layerInfo, neurons, inputs, weights, vbias);

    n = neurons;
    in = inputs;
    depth = 1;
    ddot = new double[n];
    out = new double[n];
    w = new double[in*n];
    bias = new double[n];

    for (int i = 0; i < in*n; ++i) { //randomly initializes weights
        w[i] = weights[i];
    }
    for (int i = 0; i < n; ++i) { //randomly initializes weights
        bias[i] = vbias[i];
    }
};

FCLayer::~FCLayer() {
    delete []bias;
    delete []ddot;
    delete []out;
    delete []w;
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
        ddot[i] = error[i];
    }
}

void FCLayer::computeError(double* result) { //delete
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


std::string FCLayer::printLayer() {

    std::stringstream ss;

    ss << "type:FCLayer|";
    ss << "inputs:" << in << "|";
    ss << "neurons:" << n << "|";

    ss << "weights:";
    long n_weights = in * n;
    for (int i = 0; i < n_weights; i++)
        ss << w[i] << ",";


    ss << "/bias:";
    for (int i = 0; i < n; i++)
        ss << bias[i] << ",";

    ss << "\n";

    std::string out = ss.str();
    ss.clear();

    return out;
}


void FCLayer::loadLayer(std::string layerInfo, int &neurons, int &inputs, std::vector<double> &weights, std::vector<double> &biases){
    std::string field, value, label, weights_str, biases_str;
    std::size_t position, position_f;


    // parse parameters
    while ((position = layerInfo.find('|')) != std::string::npos) {

        field = layerInfo.substr(0, position);
        layerInfo = layerInfo.substr(++position);

        // parse field
        position_f = field.find(':');

        label = field.substr(0, position_f);
        value = field.substr(++position_f);

        if (!label.compare("neurons")) {
            neurons = std::stoi(value);
        }
        else if (!label.compare("inputs")) {
            inputs = std::stoi(value);
        }
    }


    // parse weights
    field = layerInfo;
    position = field.find('/');
    weights_str = field.substr(0, position);
    biases_str = field.substr(++position);

    // weights
    position_f = weights_str.find(':');
    label = weights_str.substr(0, position_f);
    value = weights_str.substr(++position_f);

    int i = 0;

    while ((position = value.find(',')) != std::string::npos && value.compare(",")) {
        label = value.substr(0, position);
        value = value.substr(++position);
        weights.push_back((double) std::stof(label));
    }

    // weights
    position_f = biases_str.find(':');
    label = biases_str.substr(0, position_f);
    value = biases_str.substr(++position_f);

    i = 0;

    while ((position = value.find(',')) != std::string::npos && value.compare(",")) {
        label = value.substr(0, position);
        value = value.substr(++position);
        biases.push_back((double) std::stof(label));
    }
}

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


