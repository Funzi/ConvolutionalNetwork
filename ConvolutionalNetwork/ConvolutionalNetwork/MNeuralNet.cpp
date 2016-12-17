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
#define BATCH_SIZE 10000
#define MOMENTUM 0.9


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
	layers->convLayer = new ConvLayer(5,1,6,32,3,net->input->values);
	layers->poolLayer = new PoolLayer(layers->convLayer);
	layers->FCLayer = new FCLayer(16*16*6,10, layers->poolLayer);

    net->out = net->layers->FCLayer->out;
    net->errors = net->layers->FCLayer->ddot;
	
}

void MNeuralNet::Init(MyNeuralNet* net, std::string logPath) {
    net->layers = (Layers *) malloc(sizeof(Layers));
    net->input = (Input *) malloc(sizeof(Input));
    net->input->values = (double *) malloc(sizeof(double) * PICTURE_SIZE * 3);

    Layers *layers = net->layers;


    //load convLayer
    std::string line, params;
    std::ifstream logfile (logPath);

    //load convLayer
    std::getline(logfile, line);
    layers->convLayer = new ConvLayer(line,net->input->values);

    std::getline(logfile, line);
    layers->poolLayer = new PoolLayer(layers->convLayer);

    std::getline(logfile, line);
    layers->FCLayer = new FCLayer(line, layers->poolLayer);

	net->out = net->layers->FCLayer->out;
	net->errors = net->layers->FCLayer->ddot;

}

void MNeuralNet::Evaluate(MyNeuralNet * net, string path)
{
    cout << "Evaluating file " << path << ", it can take about two minutes..." << endl;
	for (int i = 0; i < BATCH_SIZE; i++) {
		EvaluateOneFile(net, path,i);
	}
	printf("\nEvaluation was correct in %f%% \n", (double)correctCount * 100 / BATCH_SIZE);
}

void MNeuralNet::EvaluateOneFile(MyNeuralNet * net, string filePath, int position)
{

	Layers* layers = net->layers;
	
	setInput(filePath, net->input, position);
	
	//printf("The result should be %d \n", ((net->input->label) + 1));
	//printf("Starting to evaluate %d file. \n", position);
	layers->convLayer->forward_layer();
	layers->poolLayer->forward_layer();
	layers->FCLayer->forward_layer();
	//layers->FCLayer->print();
	checkAnswer(net, net->input->label);
}

void MNeuralNet::Learn(MyNeuralNet* net, string path)
{
    cout << "Learning file: " <<  path << endl;
    double avgError = 0;
	for (int i = 0; i < BATCH_SIZE; i++) {
		LearnOneFile(net, path, i);
        avgError += net->totalError;
        if ((i+1) % 1000 == 0) {
            cout << "error: " << avgError/1000 << " | correct: " << correctCount/10 << "%"<< std::endl;
            avgError = 0;
            correctCount = 0;
            //net->layers->FCLayer->print();
        }
	}
    cout << "File was learned." << endl;
    cout << "Neural Net was saved in cnn.log ." << endl;
	//printf("\nEvaluation was correct in %f%% \n", (double)correctCount *100 / (double)BATCH_SIZE);
    correctCount = 0;
}

void MNeuralNet::checkAnswer(MyNeuralNet* net, int label) {
    bool ans = true;

	for (int i = 0; i < 10; i++) {
        if (net->out[i] > net->out[label]) {ans = false; break; }
    }
    if (ans) correctCount++;
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


void MNeuralNet::LearnOneFile(MyNeuralNet* net, std::string filePath, int position)
{
	Layers* layers = net->layers;
	
	setInput(filePath, net->input, position);

	//printf("Starting to learn %d.file \n", position + 1);
	//printf("The result should be %d \n", net->input->label + 1);

	layers->convLayer->forward_layer();
	layers->poolLayer->forward_layer();
	layers->FCLayer->forward_layer();
    checkAnswer(net, net->input->label);

	computeError(net);

	layers->FCLayer->backProp_layer();
	layers->poolLayer->backProp_layer();
	layers->convLayer->backProp_layer();
	
	//addError
	layers->FCLayer->learn();
	layers->poolLayer->learn();
	layers->convLayer->learn();

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





