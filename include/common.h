#ifndef COMMON_H
#define COMMON_H

#include <opencv2/core.hpp>

#define OBJ_NUMB_MAX_SIZE 64
#define NET_INPUTCHANNEL 3

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct _DetectionBox
{
    _DetectionBox(int x=0, int y=0, int width=0, int height=0, 
              float confidence=0.0, float classID=-1, float trackID=-1)
              : box(x, y, width, height), score(confidence), classID(classID), trackID(trackID), det_name("")
    {

    }
    float score;
    std::string det_name;
    cv::Rect_<int> box;
    float classID;
    float trackID;
} DetectionBox;

typedef struct _DetectResultsGroup
{
    cv::Mat cur_img;
    int cur_frame_id;
    std::vector<DetectionBox> dets; // 修改为vector
} DetectResultsGroup;

typedef struct _TrackingBox
{
    int id;
    std::string det_name;
    cv::Rect_<float> box;
} TrackingBox;

#endif // COMMON_H