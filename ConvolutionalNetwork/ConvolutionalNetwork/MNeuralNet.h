#pragma once
#include "Layer.h"
#include "ConvLayer.h"
#include "PoolLayer.h"
#include "FCLayer.h"
#include <vector>
#include <string>

#define DIM 32
#define DIM_SQR 1024

struct Layers {
	Layer* fcLayer;
	Layer* convLayer;
	Layer* poolLayer;

};

struct Input {
	int label;
	double *values;
};

struct MyNeuralNet {
	Layers* layers;
	Input* input;
	double results[10];
    double totalError;
    double* out;
    double* errors;
};

namespace MNeuralNet {
	void Init(MyNeuralNet* net);

	void Evaluate(MyNeuralNet* net, std::string path);

	void EvaluateOneFile(MyNeuralNet* net, std::string filePath, int position);

	void Learn(MyNeuralNet* net, std::string path);
	
	void LearnOneFile(MyNeuralNet* net, std::string filePath, int position);

    bool checkAnswer(MyNeuralNet* net, int label);

    void computeError(MyNeuralNet* net);

	void Release(MyNeuralNet* net);
}

