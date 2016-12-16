#include "MNeuralNet.h"
#include "FCLayer.h"
#include "ConvLayer.h"
#include "PoolLayer.h"
#include <iostream>
#include <fstream>

using namespace std;
static int correctCount = 0;

#define PICTURE_SIZE 1024
#define LABEL_SIZE 1
#define BATCH_SIZE 100

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

void createResultArray(int label, double* resultArray) {
	for (int i = 0; i < 10; i++) {
		if (i == label) {
			resultArray[i] = 1;
		}
		else {
			resultArray[i] = 0;
		}
	}
}

void setInputAndResult(std::string filename, MyNeuralNet* net, int position) {
	setInput(filename, net->input, position);
	createResultArray(net->input->label, net->results);
}

int findTheBiggestPossibility(double* output) {
	double temp = output[0];
	int biggest = 1;
	for (int i = 1; i < 10; i++) {
		if (output[i] > temp) {
			temp = output[i];
			biggest = i;
		}
	}
	return biggest;
}

void checkResult(MyNeuralNet* net) {
	double temp = findTheBiggestPossibility(net->results);
	if (temp == net->input->label)
		correctCount++;
}

void MNeuralNet::Init(MyNeuralNet* net)
{
	net->layers = (Layers*)malloc(sizeof(Layers));
	net->input = (Input*)malloc(sizeof(Input));
	net->input->values = (double*)malloc(sizeof(double) * PICTURE_SIZE * 3);
	

	Layers* layers = net->layers;
	
	// not sure what parameters to put after it I can refactor it 
	layers->convLayer = new ConvLayer(5,1,3,32,3,net->input->values);
	layers->poolLayer = new PoolLayer(layers->convLayer);
	layers->FCLayer = new FCLayer(16*16*3,10, layers->poolLayer);
	
}

void MNeuralNet::Evaluate(MyNeuralNet * net, string path)
{
	for (int i = 0; i < BATCH_SIZE; i++) {
		EvaluateOneFile(net, path,i);
	}
	printf("\nEvaluation was correct in %f%% \n", (double)correctCount / (double)BATCH_SIZE);
}

void MNeuralNet::EvaluateOneFile(MyNeuralNet * net, string filePath, int position)
{
	Layers* layers = net->layers;
	
	setInput(filePath, net->input, position);
	
	printf("The result should be %d \n", net->input->label + 1);
	printf("Starting to evaluate %d file. \n", position);
	layers->convLayer->forward_layer();
	layers->poolLayer->forward_layer();
	layers->FCLayer->forward_layer();
	layers->FCLayer->print();
	checkResult(net);
}

void MNeuralNet::Learn(MyNeuralNet* net, string path)
{
	for (int i = 0; i < BATCH_SIZE; i++) {
		LearnOneFile(net, path, i);
	}
	printf("\nEvaluation was correct in %f%% \n", (double)correctCount *100 / (double)BATCH_SIZE);

}

void MNeuralNet::LearnOneFile(MyNeuralNet* net, std::string filePath, int position)
{
	Layers* layers = net->layers;
	
	setInputAndResult(filePath, net,position);
	
	printf("Starting to learn %d.file \n", position + 1);
	printf("The result should be %d \n", net->input->label + 1);

	layers->convLayer->forward_layer();
	layers->poolLayer->forward_layer();
	layers->FCLayer->forward_layer();
	layers->FCLayer->print();
	((FCLayer*)(layers->FCLayer))->computeError(net->results);
	((FCLayer*)(layers->FCLayer))->setResults(net->results);
	checkResult(net);
	
	layers->FCLayer->backProp_layer();
	layers->poolLayer->backProp_layer();
	
	//addError
	layers->convLayer->learn();
	layers->poolLayer->learn();
	layers->FCLayer->learn();
}

void MNeuralNet::Release(MyNeuralNet* net)
{
	free(net->input->values);
	free(net->input);
}

void MNeuralNet::SaveWeights(MyNeuralNet *net, std::string filePath) {

    Layers* layers = net->layers;

    std::ofstream logfile (filePath);

    logfile << layers->convLayer->printLayer();
    logfile << layers->poolLayer->printLayer();
    logfile << layers->FCLayer->printLayer();

    logfile.close();

}





