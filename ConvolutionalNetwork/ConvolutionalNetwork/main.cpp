#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string.h>
#include <fstream>
#include "MNeuralNet.h"



#define TRAINING_SET1 "data_batch_1.bin"
#define TRAINING_SET2 "data_batch_2.bin"
#define TRAINING_SET3 "data_batch_3.bin"
#define TRAINING_SET4 "data_batch_4.bin"
#define TEST_SET "test_batch.bin"


const int LAYERS = 2; // number of FC layers including output
const int NEURONS = 2; //number of neurons in one layer
//const int ITERATIONS = 10000; //number of learning cycles

void help(){
    std::cout << "NN_Neuron is convolutional neural network for CIFAR-10 image classification. Use as follows:" << "\n\n";
    std::cout << "  MODE -m" << "\n\n";
    std::cout << "      Mode in which NN_Neuron runs, options are:" << "\n\n";
    std::cout << "      train" << "\n\n";
    std::cout << "          NN_neuron uses passed data file to train the neural network. The network is then saved into log file" << "\n\n";
    std::cout << "      trainOn" << "\n\n";
    std::cout << "          NN_neuron uses passed data file to train the neural network. The network is initialized from and then saved into log file" << "\n\n";
    std::cout << "      eval" << "\n\n";
    std::cout << "          NN_neuron initializes neural network from the log file and evaluates data file" << "\n\n";
    std::cout << "  PATH TO DATA FILE -f" << "\n\n";
    std::cout << "      Path to CIFAR-10 data file" << "\n\n";
    std::cout << "  PATH TO DATA FILE -l" << "\n\n";
    std::cout << "      Path to log file" << "\n\n";
    std::cout << "For example:" << "\n\n";
    std::cout << "./NN_Neuron -m train -f data_batch_1.bin -l cnn.log"  << "\n\n";

}


void parseCommandline(int argc, char *argv[], std::string &mode, std::string &filePath, std::string &logpath){

    bool wannaHelp = false;

    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "-h"))
            wannaHelp = true;

        if (!strcmp(argv[i], "-m") && argc > i) {
            mode = argv[i + 1];
        }
        if (!strcmp(argv[i], "-f") && argc > i)
            filePath = argv[i+1];

        if (!strcmp(argv[i], "-l") && argc > i)
            logpath = argv[i+1];
    }


    if (argc < 2 || wannaHelp)
        help();

}

int main(int argc, char *argv[]){

    std::string mode, filePath, logPath;

    parseCommandline(argc, argv, mode, filePath, logPath);

    MyNeuralNet net;
	
	if (!strcmp(mode.c_str(),"train") && !filePath.empty() && !logPath.empty()){
        MNeuralNet::Init(&net);
        MNeuralNet::Learn(&net, filePath);
        MNeuralNet::SaveWeights(&net, logPath);
    } else if (!strcmp(mode.c_str(),"trainOn") && !filePath.empty() && !logPath.empty()){
        MNeuralNet::Init(&net, logPath);
        MNeuralNet::Learn(&net, filePath);
        MNeuralNet::SaveWeights(&net, logPath);
    } else if (!strcmp(mode.c_str(),"eval") && !filePath.empty() && !logPath.empty()){
        MNeuralNet::Init(&net, logPath);
        MNeuralNet::Evaluate(&net, filePath);
    }

    getchar();
    return 0;
    
}

