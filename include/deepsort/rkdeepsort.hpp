#ifndef RKDEEPSORT_H
#define RKDEEPSORT_H

#include <condition_variable>
#include "rkYolov5s.hpp"
#include "common.h"
#include "tracker.h"
#include "featuretensor.h"
#include <thread>

#include "opencv2/core/core.hpp"

class rkDeepSort
{
public:
    rkDeepSort(const std::string &model_path);
    ~rkDeepSort();
private:
    std::string model_path;
    float confThres;
    float nmsThres;
    int featureDim;
    int maxBudget;
    float maxCosineDist;
    const int track_interval = 1;
    cv::Size imgShape;
private:
    void sort(cv::Mat& frame, DETECTIONS& detections);
    void sort(cv::Mat& frame, DETECTIONSV2& detectionsv2);  
    void init();
private:
    tracker* objTracker;
    FeatureTensor* featureExtractor1;
    FeatureTensor* featureExtractor2;

    vector<RESULT_DATA> result;
    vector<std::pair<CLSCONF, DETECTBOX>> results;
public:
    void sort(cv::Mat& frame, vector<DetectionBox>& dets);
    void sort_interval(cv::Mat& frame, vector<DetectionBox>& dets);
    int  track_process();
    void showDetection(DetectResultsGroup detsgroup);

};

#endif