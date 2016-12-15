#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include "NeuralNet.h"
#include "MNeuralNet.h"


#define TRAINING_SET1 "data_batch_1.bin"
#define TRAINING_SET2 "data_batch_2.bin"
#define TRAINING_SET3 "data_batch_3.bin"
#define TRAINING_SET4 "data_batch_4.bin"
#define TEST_SET "test_batch.bin"


int main(){
	MyNeuralNet net;
	
	MNeuralNet::Init(&net);
    for ( int i = 1; i < 16; i++) {
        std::cout << "learning " << i << std::endl;
        MNeuralNet::Learn(&net, TRAINING_SET1);
        MNeuralNet::Learn(&net, TRAINING_SET2);
        MNeuralNet::Learn(&net, TRAINING_SET3);
        MNeuralNet::Learn(&net, TRAINING_SET4);
    }
    MNeuralNet::Evaluate(&net,TEST_SET);
    
	getchar();
    return 0;
    
}

