#pragma once
#include "Layer.h"
#include <vector>
#include <string>

#define DIM 32
#define DIM_SQR 1024

struct Layers {
	Layer* FCLayer;
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

    void Init(MyNeuralNet* net, std::string logPath);

	void Evaluate(MyNeuralNet* net, std::string path);

	void EvaluateOneFile(MyNeuralNet* net, std::string filePath, int position);

	void Learn(MyNeuralNet* net, std::string path);
	
	void LearnOneFile(MyNeuralNet* net, std::string filePath, int position);

	void Release(MyNeuralNet* net);

	void SaveWeights(MyNeuralNet* net, std::string filePath);

	void checkAnswer(MyNeuralNet* net, int label);

	void computeError(MyNeuralNet* net);


}

