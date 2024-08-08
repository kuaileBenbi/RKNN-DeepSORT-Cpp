
#include "rkdeepsort.hpp"

extern std::queue<DetectResultsGroup> trackingQueue;
extern std::mutex queueMutex;
extern std::condition_variable queueCondVar;
extern bool running;

rkDeepSort::rkDeepSort(const std::string &model_path)
{
    this->model_path = model_path;
    this->featureDim = 512;
    this->imgShape = cv::Size(64, 128);
    // this->imgShape = cv::Size(128, 256);
    this->maxBudget = 100;
    this->maxCosineDist = 0.2;
    init();
}

void rkDeepSort::init()
{
    objTracker = new tracker(maxCosineDist, maxBudget);

    // two Re-ID networks, share same CPU and NPU
    featureExtractor1 = new FeatureTensor(model_path.c_str());
    featureExtractor1->init(imgShape, featureDim);

    featureExtractor2 = new FeatureTensor(model_path.c_str());
    featureExtractor2->init(imgShape, featureDim);
}

rkDeepSort::~rkDeepSort()
{
    delete objTracker;
}

void rkDeepSort::sort(cv::Mat &frame, vector<DetectionBox> &dets)
{
    // preprocess Mat -> DETECTION
    DETECTIONS detections; // DETECTIONS: std::vector<DETECTION_ROW> in model.hpp
    vector<CLSCONF> clsConf;

    // read every detections in current frame and
    // store them in detections(bbox) and clsConf(conf scores)
    for (DetectionBox i : dets)
    {
        DETECTBOX box(i.box.x, i.box.y, i.box.width, i.box.height);
        DETECTION_ROW d;
        d.tlwh = box;
        d.score = i.score;
        detections.push_back(d);
        clsConf.push_back(CLSCONF((int)i.classID, i.score));
    }

    result.clear();  // result: vector<pair<int, DETECTBOX>>
    results.clear(); // results: vector<pair<CLSCONF, DETECTBOX>>
    if (detections.size() > 0)
    {
        DETECTIONSV2 detectionsv2 = make_pair(clsConf, detections);
        sort(frame, detectionsv2); // sort
    }
    // postprocess DETECTION -> Mat
    dets.clear();
    for (auto r : result)
    {
        DETECTBOX i = r.second;
        DetectionBox b(i(0), i(1), i(2), i(3), 1.);
        b.trackID = (float)r.first;
        dets.push_back(b);
    }
    for (int i = 0; i < results.size(); ++i)
    {
        CLSCONF c = results[i].first;
        dets[i].classID = c.cls;
        dets[i].score = c.conf;
    }
}

void rkDeepSort::sort(cv::Mat &frame, DETECTIONS &detections)
{
    bool flag = featureExtractor1->getRectsFeature(frame, detections);
    if (flag)
    {
        objTracker->predict();
        objTracker->update(detections);
        // result.clear();
        for (Track &track : objTracker->tracks)
        {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
        }
    }
}

void rkDeepSort::sort_interval(cv::Mat &frame, vector<DetectionBox> &dets)
{
    /*
    If frame_id % this->track_interval != 0, there is no new detections
    so only predict the tracks using Kalman
    */
    if (!dets.empty())
        cout << "Error occured! \n";

    result.clear();
    results.clear();
    objTracker->predict(); // Kalman predict

    // update result and results
    // cout << "---------" << objTracker->tracks.size() << "\n";
    for (Track &track : objTracker->tracks)
    {
        // if (!track.is_confirmed() || track.time_since_update > 1)
        if (!track.is_confirmed() || track.time_since_update > this->track_interval + 1)
            continue;
        result.push_back(make_pair(track.track_id, track.to_tlwh()));
        results.push_back(make_pair(CLSCONF(track.cls, track.conf), track.to_tlwh()));
    }
    dets.clear();
    for (auto r : result)
    {
        DETECTBOX i = r.second;
        DetectionBox b(i(0), i(1), i(2), i(3), 1.);
        b.trackID = (float)r.first;
        dets.push_back(b);
    }
    for (int i = 0; i < results.size(); ++i)
    {
        CLSCONF c = results[i].first;
        dets[i].classID = c.cls;
        dets[i].score = c.conf;
    }
}

void rkDeepSort::sort(cv::Mat &frame, DETECTIONSV2 &detectionsv2)
{
    std::vector<CLSCONF> &clsConf = detectionsv2.first;
    DETECTIONS &detections = detectionsv2.second; // std::vector<DETECTION_ROW>

    int numOfDetections = detections.size();
    bool flag1 = true, flag2 = true;
    if (numOfDetections < 2)
    {
        // few objects, use single Re-ID
        // double timeBeforeReID = what_time_is_it_now();
        flag1 = featureExtractor1->getRectsFeature(frame, detections);
        // double timeAfterReID = what_time_is_it_now();

        // cout << "--------Time cost in ReID: " << timeAfterReID - timeBeforeReID << "\n";
        flag2 = true;
    }
    else
    {
        DETECTIONS detectionsPart1, detectionsPart2;
        int border = numOfDetections >> 1;
        auto start = detections.begin(), end = detections.end(); // iterator

        // double timeBeforeAssign = what_time_is_it_now();
        detectionsPart1.assign(start, start + border);
        detectionsPart2.assign(start + border, end);
        // double timeAfterAssign = what_time_is_it_now();

        // cout << "--------Time cost in assign: " << timeAfterAssign - timeBeforeAssign << "\n";

        // NOTE: convert pointer or set global variables
        // inference separately
        // double timeBeforeReID = what_time_is_it_now();
        thread reID1Thread1(&FeatureTensor::getRectsFeature, featureExtractor1, std::ref(frame), std::ref(detectionsPart1));
        thread reID1Thread2(&FeatureTensor::getRectsFeature, featureExtractor2, std::ref(frame), std::ref(detectionsPart2));

        reID1Thread1.join();
        reID1Thread2.join();

        // double timeAfterReID = what_time_is_it_now();

        // cout << "--------Time cost in ReID: " << timeAfterReID - timeBeforeReID << "\n";

        // copy new feature to origin detections

        // double timeBeforeUpdateFeatures = what_time_is_it_now();
        for (int idx = 0; flag1 && flag2 && idx < numOfDetections; idx++)
        {
            if (idx < border)
                detections[idx].updateFeature(detectionsPart1[idx].feature);
            else
                detections[idx].updateFeature(detectionsPart2[idx - border].feature);
        }
        // double timeAfterUpdateFeatures = what_time_is_it_now();
        // cout << "--------Time cost in update features: " << timeAfterUpdateFeatures - timeBeforeUpdateFeatures << "\n";
    }

    // bool flag = featureExtractor->getRectsFeature(frame, detections);
    if (flag1 && flag2)
    {
        objTracker->predict();
        // std::cout << "In: \n";
        objTracker->update(detectionsv2);
        // std::cout << "Out: \n";
        result.clear();
        results.clear();
        for (Track &track : objTracker->tracks)
        {
            if (!track.is_confirmed() || track.time_since_update > track._max_age)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
            results.push_back(make_pair(CLSCONF(track.cls, track.conf), track.to_tlwh()));
        }
    }
    else
        cout << "Re-ID1 Error? " << flag1 << " Re-ID2 Error? " << flag2 << "\n";
}

int rkDeepSort::track_process()
{
    while (running || !trackingQueue.empty())
    {
        DetectResultsGroup curDetectRes;

        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCondVar.wait(lock, []
                              { return !trackingQueue.empty() || !running; });

            if (!trackingQueue.empty())
            {
                curDetectRes = trackingQueue.front();
                trackingQueue.pop();
                queueCondVar.notify_one();  // 添加通知
            }
        }

        int curFrameIdx = curDetectRes.cur_frame_id;

        if (curFrameIdx < this->track_interval || !(curFrameIdx % this->track_interval)) // have detections
        {
            sort(curDetectRes.cur_img, curDetectRes.dets); // 会更新 dets.results
            showDetection(curDetectRes);
        }

        else
        {
            sort_interval(curDetectRes.cur_img, curDetectRes.dets);
            showDetection(curDetectRes);
        }

        // mtxQueueOutput.lock();
        // // cout << "--------------" << queueDetOut.front().dets.results.size() << "\n";
        // queueOutput.push(queueDetOut.front());
        // mtxQueueOutput.unlock();
        // // showDetection(queueDetOut.front().img, queueDetOut.front().dets.results);
        // mtxQueueDetOut.lock();
        // queueDetOut.pop();
        // mtxQueueDetOut.unlock();
    }
    cout << "Track is over." << endl;
    return 0;
}

void rkDeepSort::showDetection(DetectResultsGroup detsgroup)
{
    cv::Mat temp = detsgroup.cur_img.clone();

    for (auto box : detsgroup.dets)
    {
        // cv::Point lt(box.box.x, box.box.x);
        // cv::Point br(box.box.x + box.box.width, box.box.y + box.box.height);
        cv::rectangle(temp, box.box, cv::Scalar(255, 0, 0), 2);
        // std::string lbl = cv::format("ID:%d_C:%d_CONF:%.2f", (int)box.trackID, (int)box.classID, box.confidence);
        // std::string lbl = cv::format("ID:%d_C:%d", (int)box.trackID, (int)box.classID);
        std::string lbl = cv::format("ID:%d", (int)box.trackID);
        cv::putText(temp, lbl, cv::Point(box.box.x, box.box.y + 12), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 255, 0));
    }
    std::ostringstream oss;

    oss << "track_" << std::setfill('0') << std::setw(4) << detsgroup.cur_frame_id << ".jpg";

    std::string filename = oss.str();
    cv::imwrite(filename, temp);
}