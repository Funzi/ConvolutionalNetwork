#include "MNeuralNet.h"
#include "FCLayer.h"
#include "ConvLayer.h"
#include "PoolLayer.h"
#include <iostream>
#include <fstream>

using namespace std;

#define PICTURE_SIZE 1024
#define LABEL_SIZE 1
#define BATCH_SIZE 10000
#define MOMENTUM 0.9

void parseWeights(std::string weights_str, std::vector<double> &weights) {

	std::string value;
	std::size_t position;

	while ((position = weights_str.find(',')) != std::string::npos && weights_str.compare(",")) {
		value = weights_str.substr(0, position);
		weights_str = weights_str.substr(++position);
		weights.push_back((double)std::stof(value));
	}
}

void parseLogLine(std::string line, int &layerCode, int &neurons, int &inputs, std::vector<double> &weights) {

	std::string field, value, label;
	std::size_t position, position_f;

	// parse parameters
	while ((position = line.find('|')) != std::string::npos) {

		field = line.substr(0, position);
		line = line.substr(++position);

		// parse field
		position_f = field.find(':');

		label = field.substr(0, position_f);
		value = field.substr(++position_f);

		if (!label.compare("layerCode")) {
			layerCode = std::stoi(value);
		}
		else if (!label.compare("neurons")) {
			neurons = std::stoi(value);
		}
	}

	// parse weights
	field = line;
	position_f = field.find(':');
	label = field.substr(0, position_f);
	value = field.substr(++position_f);
	parseWeights(value, weights);

}

static void setInput(std::string filename, Input* input, int position)
{
	// open file
	std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);
	file.seekg((LABEL_SIZE + PICTURE_SIZE * 3)*position);
	
	char tempLabel;
	char red[1024];
	char green[1024];
	char blue[1024];

	// get label and each of these channels
	file.get(tempLabel);
	file.read(red, PICTURE_SIZE);
	file.read(green, PICTURE_SIZE);
	file.read(blue, PICTURE_SIZE);
	
	input->label = (int)tempLabel;

	for (int i = 0; i < PICTURE_SIZE; i++) {
		input->values[i] = (double)red[i];
	}
	for (int i = PICTURE_SIZE; i < PICTURE_SIZE * 2; i++) {
		input->values[i] = (double)green[i];
	}
	for (int i = PICTURE_SIZE * 2; i < PICTURE_SIZE * 3; i++) {
		input->values[i] = (double)blue[i];
	}
	file.close();
}


void MNeuralNet::Init(MyNeuralNet* net)
{
	net->layers = (Layers*)malloc(sizeof(Layers));
	net->input = (Input*)malloc(sizeof(Input));
	net->input->values = (double*)malloc(sizeof(double) * PICTURE_SIZE * 3);

	Layers* layers = net->layers;


	// not sure what parameters to put after it I can refactor it 
	layers->convLayer = new ConvLayer(5,1,12,32,3,net->input->values);
	layers->poolLayer = new PoolLayer(layers->convLayer);
	layers->fcLayer = new FCLayer(16*16*12,10, layers->poolLayer);
    layers->convLayer->input = net->input->values;

    net->out = net->layers->fcLayer->out;
	net->errors = net->layers->fcLayer->ddot;
}

void MNeuralNet::Evaluate(MyNeuralNet * net, string path)
{
    cout << "Evaluating net..." << endl;
    int correct = 0;
	for (int i = 0; i < BATCH_SIZE; i++) {
		EvaluateOneFile(net, path,i);
        if (checkAnswer(net, net->input->label)) correct++;
	}
    cout << "Number of correct answers: " << correct << " -> " << correct / 100 << '%' <<endl;
}

void MNeuralNet::EvaluateOneFile(MyNeuralNet * net, string filePath, int position)
{
	Layers* layers = net->layers;
	
	setInput(filePath, net->input, position);

	layers->convLayer->forward_layer();
	layers->poolLayer->forward_layer();
	layers->fcLayer->forward_layer();
}

void MNeuralNet::Learn(MyNeuralNet* net, string path)
{
    double avgError = 0;
    int correct = 0;
	for (int i = 0; i < 10000; i++) {
		LearnOneFile(net, path, i);
        avgError += net->totalError;
        if (checkAnswer(net, net->input->label)) correct++;
        if ((i+1) % 1000 == 0) {
            cout << "error: " << avgError/1000 << " | correct: " << correct << std::endl;
            avgError = 0;
            correct = 0;
            net->layers->fcLayer->print();
        }
	}
}


void MNeuralNet::computeError(MyNeuralNet* net) {
    double sum = 0;
	for (int i = 0; i < 10; i++) {
        net->errors[i] *= 1- MOMENTUM;
        if (i == net->input->label)
			net->errors[i] += (net->out[i] - 1)*MOMENTUM;
        else net->errors[i] += net->out[i]*MOMENTUM;
		sum += pow(net->errors[i], 2);
    }
    net->totalError = sum/2;
}



bool MNeuralNet::checkAnswer(MyNeuralNet* net, int label) {
        for (int i = 0; i < 10; i++) {
            if (net->out[i] > net->out[label]) return false;
        }
        return true;
}

void MNeuralNet::LearnOneFile(MyNeuralNet* net, std::string filePath, int position) {
	Layers* layers = net->layers;
	
	setInput(filePath, net->input,position);

	layers->convLayer->forward_layer();
	layers->poolLayer->forward_layer();
	layers->fcLayer->forward_layer();
    computeError(net);

	layers->fcLayer->backProp_layer();
	layers->poolLayer->backProp_layer();
	layers->convLayer->backProp_layer();

	layers->fcLayer->learn();
	layers->poolLayer->learn();
	layers->convLayer->learn();
}

void MNeuralNet::Release(MyNeuralNet* net)
{
	free(net->input->values);
	free(net->input);
}



