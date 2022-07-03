//
// Created by linhan on 2021/2/2.
//
#include "crnn.h"


namespace OcrSystem
{
void CrnnRec::paramInit(std::string configPath)
{
    std::string configFile = configPath + "/ocr.yaml";
    cv::FileStorage fs(configFile, cv::FileStorage::READ);
    BATCH_SIZE = fs["BATCH_SIZE"];
    INPUT_H = fs["INPUT_H"];
    INPUT_W = fs["INPUT_W"];
    OUTPUT_H = fs["OUTPUT_H"];
    OUTPUT_W = fs["OUTPUT_W"];
    STD = fs["STD"];
    MEAN = fs["MEAN"];
    OUTPUT_SIZE = OUTPUT_H * OUTPUT_W;
    INPUT_BLOB_NAME = "data";
    OUTPUT_BLOB_NAME = "prob";
    alphabet = (std::string)fs["alphabet"];
    engineFile = (std::string)fs["engine_flile"];
    size = {0};
    trtModelStream = nullptr;
    inputDataVec = std::vector<float>(BATCH_SIZE * 1 * INPUT_H * INPUT_W);
    inputData = inputDataVec.data();
    probVec = std::vector<float>(BATCH_SIZE * OUTPUT_SIZE);
    prob = probVec.data();
}

CrnnRec::CrnnRec(std::string configPath)
{
    paramInit(configPath);
    std::string paramsFile = configPath + "/" + engineFile;
    int initStatus = crnnInitModel(paramsFile);
    if(initStatus != 0)
    {
        std::cerr << "Crnn model initialization failed " << std::endl;
    }
}

std::string CrnnRec::strDecode(std::vector<int>& preds, bool raw)
{
    std::string str;
    if (raw)
    {
        for (auto v: preds)
        {
            //str.push_back(alphabet[v]);
            if(v == 0)
            {
                str += wordDic[v];
            }
            else
            {
                str += wordDic[v - 1];
            }
        }
    }
    else
    {
        for (size_t i = 0; i < preds.size(); i++)
        {
            if (preds[i] == 0 || (i > 0 && preds[i - 1] == preds[i])) continue;
            str += wordDic[preds[i] -1];
        }
    }
    return str;
}

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, nvinfer1::Weights> CrnnRec::loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, nvinfer1::Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = nvinfer1::DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

nvinfer1::IScaleLayer* CrnnRec::addBatchNorm2d(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, std::string lname, float eps)
{
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        pval[i] = 1.0;
    }
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    nvinfer1::IScaleLayer* scale_1 = network->addScale(input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

nvinfer1::ILayer* CrnnRec::convRelu(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int i, bool use_bn)
{
    int nOut = nm[i];
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(input, nOut, nvinfer1::DimsHW{ks[i], ks[i]}, weightMap["cnn.conv" + std::to_string(i) + ".weight"], weightMap["cnn.conv" + std::to_string(i) + ".bias"]);
    assert(conv);
    conv->setStrideNd(nvinfer1::DimsHW{ss[i], ss[i]});
    conv->setPaddingNd(nvinfer1::DimsHW{ps[i], ps[i]});
    nvinfer1::ILayer *tmp = conv;
    if (use_bn)
    {
        tmp = addBatchNorm2d(network, weightMap, *conv->getOutput(0), "cnn.batchnorm" + std::to_string(i), 1e-5);
    }
    auto relu = network->addActivation(*tmp->getOutput(0), nvinfer1::ActivationType::kRELU);
    assert(relu);
    return relu;
}

void CrnnRec::splitLstmWeights(std::map<std::string, nvinfer1::Weights>& weightMap, std::string lname)
{
    int weight_size = weightMap[lname].count;
    for (int i = 0; i < 4; i++)
    {
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        wt.count = weight_size / 4;
        float *val = reinterpret_cast<float*>(malloc(sizeof(float) * wt.count));
        memcpy(val, (float*)weightMap[lname].values + wt.count * i, sizeof(float) * wt.count);
        wt.values = val;
        weightMap[lname + std::to_string(i)] = wt;
    }
}

nvinfer1::ILayer* CrnnRec::addLSTM(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int nHidden, std::string lname)
{
    splitLstmWeights(weightMap, lname + ".weight_ih_l0");
    splitLstmWeights(weightMap, lname + ".weight_hh_l0");
    splitLstmWeights(weightMap, lname + ".bias_ih_l0");
    splitLstmWeights(weightMap, lname + ".bias_hh_l0");
    splitLstmWeights(weightMap, lname + ".weight_ih_l0_reverse");
    splitLstmWeights(weightMap, lname + ".weight_hh_l0_reverse");
    splitLstmWeights(weightMap, lname + ".bias_ih_l0_reverse");
    splitLstmWeights(weightMap, lname + ".bias_hh_l0_reverse");
    nvinfer1::Dims dims = input.getDimensions();
    std::cout << "lstm input shape: " << dims.nbDims << " [" << dims.d[0] << " " << dims.d[1] << " " << dims.d[2] << "]"<< std::endl;
    auto lstm = network->addRNNv2(input, 1, nHidden, dims.d[1], nvinfer1::RNNOperation::kLSTM);
    lstm->setDirection(nvinfer1::RNNDirection::kBIDIRECTION);
    lstm->setWeightsForGate(0, nvinfer1::RNNGateType::kINPUT, true, weightMap[lname + ".weight_ih_l00"]);
    lstm->setWeightsForGate(0, nvinfer1::RNNGateType::kFORGET, true, weightMap[lname + ".weight_ih_l01"]);
    lstm->setWeightsForGate(0, nvinfer1::RNNGateType::kCELL, true, weightMap[lname + ".weight_ih_l02"]);
    lstm->setWeightsForGate(0, nvinfer1::RNNGateType::kOUTPUT, true, weightMap[lname + ".weight_ih_l03"]);

    lstm->setWeightsForGate(0, nvinfer1::RNNGateType::kINPUT, false, weightMap[lname + ".weight_hh_l00"]);
    lstm->setWeightsForGate(0, nvinfer1::RNNGateType::kFORGET, false, weightMap[lname + ".weight_hh_l01"]);
    lstm->setWeightsForGate(0, nvinfer1::RNNGateType::kCELL, false, weightMap[lname + ".weight_hh_l02"]);
    lstm->setWeightsForGate(0, nvinfer1::RNNGateType::kOUTPUT, false, weightMap[lname + ".weight_hh_l03"]);

    lstm->setBiasForGate(0, nvinfer1::RNNGateType::kINPUT, true, weightMap[lname + ".bias_ih_l00"]);
    lstm->setBiasForGate(0, nvinfer1::RNNGateType::kFORGET, true, weightMap[lname + ".bias_ih_l01"]);
    lstm->setBiasForGate(0, nvinfer1::RNNGateType::kCELL, true, weightMap[lname + ".bias_ih_l02"]);
    lstm->setBiasForGate(0, nvinfer1::RNNGateType::kOUTPUT, true, weightMap[lname + ".bias_ih_l03"]);

    lstm->setBiasForGate(0, nvinfer1::RNNGateType::kINPUT, false, weightMap[lname + ".bias_hh_l00"]);
    lstm->setBiasForGate(0, nvinfer1::RNNGateType::kFORGET, false, weightMap[lname + ".bias_hh_l01"]);
    lstm->setBiasForGate(0, nvinfer1::RNNGateType::kCELL, false, weightMap[lname + ".bias_hh_l02"]);
    lstm->setBiasForGate(0, nvinfer1::RNNGateType::kOUTPUT, false, weightMap[lname + ".bias_hh_l03"]);

    lstm->setWeightsForGate(1, nvinfer1::RNNGateType::kINPUT, true, weightMap[lname + ".weight_ih_l0_reverse0"]);
    lstm->setWeightsForGate(1, nvinfer1::RNNGateType::kFORGET, true, weightMap[lname + ".weight_ih_l0_reverse1"]);
    lstm->setWeightsForGate(1, nvinfer1::RNNGateType::kCELL, true, weightMap[lname + ".weight_ih_l0_reverse2"]);
    lstm->setWeightsForGate(1, nvinfer1::RNNGateType::kOUTPUT, true, weightMap[lname + ".weight_ih_l0_reverse3"]);

    lstm->setWeightsForGate(1, nvinfer1::RNNGateType::kINPUT, false, weightMap[lname + ".weight_hh_l0_reverse0"]);
    lstm->setWeightsForGate(1, nvinfer1::RNNGateType::kFORGET, false, weightMap[lname + ".weight_hh_l0_reverse1"]);
    lstm->setWeightsForGate(1, nvinfer1::RNNGateType::kCELL, false, weightMap[lname + ".weight_hh_l0_reverse2"]);
    lstm->setWeightsForGate(1, nvinfer1::RNNGateType::kOUTPUT, false, weightMap[lname + ".weight_hh_l0_reverse3"]);

    lstm->setBiasForGate(1, nvinfer1::RNNGateType::kINPUT, true, weightMap[lname + ".bias_ih_l0_reverse0"]);
    lstm->setBiasForGate(1, nvinfer1::RNNGateType::kFORGET, true, weightMap[lname + ".bias_ih_l0_reverse1"]);
    lstm->setBiasForGate(1, nvinfer1::RNNGateType::kCELL, true, weightMap[lname + ".bias_ih_l0_reverse2"]);
    lstm->setBiasForGate(1, nvinfer1::RNNGateType::kOUTPUT, true, weightMap[lname + ".bias_ih_l0_reverse3"]);

    lstm->setBiasForGate(1, nvinfer1::RNNGateType::kINPUT, false, weightMap[lname + ".bias_hh_l0_reverse0"]);
    lstm->setBiasForGate(1, nvinfer1::RNNGateType::kFORGET, false, weightMap[lname + ".bias_hh_l0_reverse1"]);
    lstm->setBiasForGate(1, nvinfer1::RNNGateType::kCELL, false, weightMap[lname + ".bias_hh_l0_reverse2"]);
    lstm->setBiasForGate(1, nvinfer1::RNNGateType::kOUTPUT, false, weightMap[lname + ".bias_hh_l0_reverse3"]);
    return lstm;
}

// Creat the engine using only the API and not any parser.
nvinfer1::ICudaEngine* CrnnRec::createEngine(std::string paramsFile, unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt)
{
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {C, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    nvinfer1::ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, nvinfer1::Dims3{1, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(paramsFile);

    // cnn
    auto x = convRelu(network, weightMap, *data, 0);
    auto p = network->addPoolingNd(*x->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
    p->setStrideNd(nvinfer1::DimsHW{2, 2});
    x = convRelu(network, weightMap, *p->getOutput(0), 1);
    p = network->addPoolingNd(*x->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
    p->setStrideNd(nvinfer1::DimsHW{2, 2});
    x = convRelu(network, weightMap, *p->getOutput(0), 2, true);
    x = convRelu(network, weightMap, *x->getOutput(0), 3);
    p = network->addPoolingNd(*x->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
    p->setStrideNd(nvinfer1::DimsHW{2, 1});
    p->setPaddingNd(nvinfer1::DimsHW{0, 1});
    x = convRelu(network, weightMap, *p->getOutput(0), 4, true);
    x = convRelu(network, weightMap, *x->getOutput(0), 5);
    p = network->addPoolingNd(*x->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
    p->setStrideNd(nvinfer1::DimsHW{2, 1});
    p->setPaddingNd(nvinfer1::DimsHW{0, 1});
    x = convRelu(network, weightMap, *p->getOutput(0), 6, true);

    auto sfl = network->addShuffle(*x->getOutput(0));
    sfl->setFirstTranspose(nvinfer1::Permutation{1, 2, 0});

    // rnn
    auto lstm0 = addLSTM(network, weightMap, *sfl->getOutput(0), 256, "rnn.0.rnn");
    auto sfl0 = network->addShuffle(*lstm0->getOutput(0));
    sfl0->setReshapeDimensions(nvinfer1::Dims4{41, 1, 1, 512});
    auto fc0 = network->addFullyConnected(*sfl0->getOutput(0), 256, weightMap["rnn.0.embedding.weight"], weightMap["rnn.0.embedding.bias"]);

    sfl = network->addShuffle(*fc0->getOutput(0));
    sfl->setFirstTranspose(nvinfer1::Permutation{2, 3, 0, 1});
    sfl->setReshapeDimensions(nvinfer1::Dims3{1, 41, 256});

    auto lstm1 = addLSTM(network, weightMap, *sfl->getOutput(0), 256, "rnn.1.rnn");
    auto sfl1 = network->addShuffle(*lstm1->getOutput(0));
    sfl1->setReshapeDimensions(nvinfer1::Dims4{41, 1, 1, 512});
    auto fc1 = network->addFullyConnected(*sfl1->getOutput(0), 70, weightMap["rnn.1.embedding.weight"], weightMap["rnn.1.embedding.bias"]);
    nvinfer1::Dims dims = fc1->getOutput(0)->getDimensions();
    std::cout << "fc1 shape " << dims.d[0] << " " << dims.d[1] << " " << dims.d[2] << std::endl;

    fc1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*fc1->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
    #ifdef USE_FP16
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    #endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void CrnnRec::APIToModel(std::string paramsFile, unsigned int maxBatchSize, nvinfer1::IHostMemory** modelStream)
{
    // Create builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    nvinfer1::ICudaEngine* engine = createEngine(paramsFile, maxBatchSize, builder, config, nvinfer1::DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void CrnnRec::doInference(nvinfer1::IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize)
{
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 1 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

void CrnnRec::structureWordDic(void)
{
    std::vector<char> tempArr;
    for (int i = 0; i < alphabet.length(); i++)
    {
        if ((alphabet[i] & 0x80) != 0)
        {
            tempArr.push_back(alphabet[i]);
            tempArr.push_back(alphabet[i+1]);
            tempArr.push_back(alphabet[i+2]);
            wordDic.push_back(std::string(tempArr.begin(), tempArr.end()));
            tempArr.clear();
            i += 2;
        }
        else
        {
            tempArr.push_back(alphabet[i]);
            wordDic.push_back(std::string(tempArr.begin(), tempArr.end()));
            tempArr.clear();
        }
    }
}

int CrnnRec::crnnInitModel(std::string paramsFile)
{
    structureWordDic();
    cudaSetDevice(DEVICE);
    int time_start_pos = paramsFile.find_last_of('/');
    std::string enginePath = paramsFile.substr(0, time_start_pos) + "/crnn.engine";
    std::cout << "enginePath:" << enginePath << std::endl;
    // create a model using the API directly and serialize it to a stream
    if (access(enginePath.c_str(), F_OK) != 0)  //engine file not find, and create engine
    {
        nvinfer1::IHostMemory* modelStream{nullptr};
        APIToModel(paramsFile, BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p(enginePath.c_str(), std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
    }
    //load create engine file
    std::ifstream file(enginePath, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 1 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    CHECK(cudaStreamCreate(&stream));
    return 0;
}

void CrnnRec::releaseResource()
{
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

int CrnnRec::getRecResult(std::vector<cv::Mat> &batchImg, std::vector<OcrSystem::BatchOcrResult> &batchOcrRes)
{
    for (int i = 0; i < batchImg.size(); i++)
    {
        if (batchImg[i].empty())
        {
            std::cerr << "img is empty not found !!!" << std::endl;
            return -1;
        }
        cv::Mat srcimg;
        cv::Mat crop_img;
        batchImg[i].copyTo(srcimg);
        for (int j = 0; j < batchOcrRes[i].size(); j++)
        {
            crop_img = srcimg(cv::Rect(batchOcrRes[i][j].top_left.x, batchOcrRes[i][j].top_left.y,
                                       batchOcrRes[i][j].bottom_right.x - batchOcrRes[i][j].top_left.x,
                                       batchOcrRes[i][j].bottom_right.y - batchOcrRes[i][j].top_left.y));
            cv::cvtColor(crop_img, crop_img, CV_BGR2GRAY);
            cv::resize(crop_img, crop_img, cv::Size(INPUT_W, INPUT_H));
            for (int i = 0; i < INPUT_H * INPUT_W; i++)
            {
                inputData[i] = ((float)crop_img.at<uchar>(i) / 255.0 - MEAN) / STD;
            }
            // Run inference
            doInference(*context, stream, buffers, inputData, prob, BATCH_SIZE);
            std::vector<int> preds;
            for (int i = 0; i < OUTPUT_H; i++)
            {
                int maxj = 0;
                for (int j = 1; j < OUTPUT_W; j++)
                {
                    if (prob[OUTPUT_W * i + j] > prob[OUTPUT_W * i + maxj]) maxj = j;
                }
                preds.push_back(maxj);
            }
            batchOcrRes[i][j].label = strDecode(preds, false);
        }
    }
    return 0;
}
}
