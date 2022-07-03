//
// Created by linhan on 2021/2/2.
//

#ifndef CRNN_H
#define CRNN_H

#include <iostream>
#include <chrono>
#include <map>
#include <fstream>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "ocrStruct.h"

namespace OcrSystem
{
#define CHECK(status) \
do\
{\
    auto ret = (status);\
    if (ret != 0)\
    {\
        std::cerr << "Cuda failure: " << ret << std::endl;\
        abort();\
    }\
} while (0)

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id


class CrnnRec
{
    public:
        CrnnRec(std::string configPath);
        void releaseResource();
        int getRecResult(std::vector<cv::Mat> &batchImg, std::vector<OcrSystem::BatchOcrResult> &batchOcrRes);
    private:
        void paramInit(std::string configFile);
        int crnnInitModel(std::string paramsFile);
        std::string strDecode(std::vector<int>& preds, bool raw);
        std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file);
        nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, std::string lname, float eps);
        nvinfer1::ILayer* convRelu(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int i, bool use_bn = false);
        void splitLstmWeights(std::map<std::string, nvinfer1::Weights>& weightMap, std::string lname);
        nvinfer1::ILayer* addLSTM(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int nHidden, std::string lname);
        nvinfer1::ICudaEngine* createEngine(std::string paramsFile, unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt);
        void APIToModel(std::string paramsFile, unsigned int maxBatchSize, nvinfer1::IHostMemory** modelStream);
        void doInference(nvinfer1::IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize);
        void structureWordDic(void);

    private:
        int BATCH_SIZE;
        int INPUT_H;
        int INPUT_W;
        int OUTPUT_H;
        int OUTPUT_W;
        int OUTPUT_SIZE;
        float MEAN;
        float STD;
        char* INPUT_BLOB_NAME;
        char* OUTPUT_BLOB_NAME;
        std::vector<float> inputDataVec;
        float *inputData;
        std::vector<float> probVec;
        float *prob;
        Logger gLogger;
        std::vector<std::string> wordDic;
        size_t size;
        char *trtModelStream;
        nvinfer1::IExecutionContext* context;
        nvinfer1::ICudaEngine* engine;
        nvinfer1::IRuntime* runtime;
        cudaStream_t stream;
        void* buffers[2];
        int inputIndex;
        int outputIndex;
        const int ks[7] = {3, 3, 3, 3, 3, 3, 2};
        const int ps[7] = {1, 1, 1, 1, 1, 1, 0};
        const int ss[7] = {1, 1, 1, 1, 1, 1, 1};
        const int nm[7] = {64, 128, 256, 256, 512, 512, 512};
        std::string alphabet;
        std::string engineFile;
};
}
#endif //CRNN_H
