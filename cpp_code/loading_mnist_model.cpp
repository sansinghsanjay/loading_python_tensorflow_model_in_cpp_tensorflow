// libraries
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string.h>
#include <vector>
#include <sstream>
#include <typeinfo>
#include <sys/time.h>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

// namespaces
using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

// global variable
Session* session;	// tensorflow session
Status status;	// tensorflow status

// function declaration
long get_prediction(Tensor a);
long long get_timeMillis();

// function definition
long get_prediction(Tensor a) {
	long prediction;
	// tensor for storing output
	vector<Tensor> outputs;
	// preparing input
	std::vector<std::pair<string, tensorflow::Tensor>> inputs = {{"x", a}};
	// getting prediction for test data
	status = session->Run(inputs, {"pred"}, {}, &outputs);
	if (!status.ok()) {
		cout<<"Error@get_prediction: "<<status.ToString()<<"\n";
		return 1l;
	}
	prediction = outputs[0].scalar<long>()(0);
	return prediction;
}

// function definition
long long get_timeMillis() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	long long ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
	return ms;
}

// main function
int main(int argc, char *argv[]) {
	// clear terminal
	system("clear");
	// variables
	ifstream f;
	string line = "";
	string token = "";
	float temp = 0.0f;
	float matches = 0.0f, accuracy = 0.0f;
	int row_no=0, col_no=0;
	long prediction = 0l, actual = 0l;
	long long start_time, time_taken;
	// initialize a tensorflow session
	status = NewSession(SessionOptions(), &session);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 1;
	}
	// tensorflow graph definitions
	GraphDef graph_def;
	// loading tensograph of pretrained model
	status = ReadBinaryProto(Env::Default(), "/home/local/ALGOANALYTICS/sanjay/tensorflow-master/tensorflow/cc/loading_mnist_model/tf_graph_ckpt/graph_ckpt.pb", &graph_def);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 1;
	}
	// adding graph to session
	status = session->Create(graph_def);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 1;
	}
	session->Run({}, {}, {"init_all_vars_op"}, nullptr);
	// creating tensors
	auto a = Tensor(DT_FLOAT, TensorShape({1, 784}));
	// opening csv file
	f.open("/home/local/ALGOANALYTICS/sanjay/tensorflow-master/tensorflow/cc/mnist_data/test_data_forCPP.csv");
	// starting timer
	start_time = get_timeMillis();
	// reading csv file
	while(getline(f, line)) {
		// creating stream of string
		istringstream iss(line);
		// splitting line and saving into token
		while(getline(iss, token, ',')) {
			// converting token into float
			temp = stof(token.c_str());
			// filling feature tensor
			if(col_no < 784)
				a.matrix<float>()(0, col_no) = temp;
			// filling label tensor
			if(col_no == 784)
				actual = (long) temp;
			// adjusting col_no
			col_no += 1;
		}
		// adjusting col_no and row_no
		col_no = 0;
		row_no += 1;
		// getting prediction
		prediction = get_prediction(a);
		// if actual and prediction matches, increment matches
		if(actual == prediction)
			matches += 1;
	}
	// getting time
	time_taken = get_timeMillis() - start_time;
	// finding accuracy
	accuracy = matches / (row_no);
	cout<<"Total Rows: "<<(row_no)<<endl;
	cout<<"Accuracy: "<<accuracy<<endl;
	cout<<"Time Taken: "<<time_taken<<endl;
	// free any resources used by the session
	session->Close();
	return 0;
}
