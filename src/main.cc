#include <stdio.h>
#include <memory>
#include <sys/time.h>

#include "rkYolov5s.hpp"
#include "rkdeepsort.hpp"
#include "rknnPool.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// 全局变量和对象
std::queue<DetectResultsGroup> trackingQueue;
std::mutex queueMutex;
std::condition_variable queueCondVar;
bool running = true;

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printf("Usage: %s <detect model path> <deepsort model path> <video path>\n", argv[0]);
        return -1;
    }

    char *detect_model = NULL;
    detect_model = (char *)argv[1];
    char *track_model = NULL;
    track_model = (char *)argv[2];
    const char *vedio_name = argv[3];

    // 初始化rknn线程池 /Initialize the rknn thread pool
    int detectThreadNum = 2;

    rknnPool<rkYolov5s, cv::Mat, DetectResultsGroup> detectPool(detect_model, detectThreadNum);

    if (detectPool.init() != 0)
    {
        printf("rknnPool init fail!\n");
        return -1;
    }
    class rkDeepSort track(track_model);
    /*目标跟踪线程在初始化跟踪器对象后立即启动 即在主循环开始处理视频帧之前就已经启动 并持续运行直到程序结束*/
    std::thread trackingThread(&rkDeepSort::track_process, &track);

    cv::VideoCapture capture;
    capture.open(vedio_name);

    if (!capture.isOpened())
    {
        printf("Error: Could not open video or camera\n");
        return -1;
    }

    struct timeval time;
    gettimeofday(&time, nullptr);
    auto startTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    int frames = 0;
    auto beforeTime = startTime;

    while (capture.isOpened())
    {
        cv::Mat img;
        DetectResultsGroup results_group;
        if (capture.read(img) == false)
        {
            printf("read original images failed or work done!\n");
            break;
        }

        if (detectPool.put(img, frames) != 0)
        // if (detectPool.put(img) != 0)
        {
            printf("put original images failed or work done!\n");
            break;
        }

        // if (frames >= threadNum && detectPool.get(img) != 0)
        if (frames >= detectThreadNum && detectPool.get(results_group) != 0)

        {
            printf("frames > 3 but get processed images failed! or work done\n");
            break;
        }
        if (!results_group.cur_img.empty())
        // 将检测结果放入跟踪队列
        {
            // cv::Mat img_tmp = results_group.cur_img.clone();
            // draw_image_detect(img_tmp, results_group.dets, results_group.cur_frame_id);
            std::lock_guard<std::mutex> lock(queueMutex);
            trackingQueue.push(results_group);
            queueCondVar.notify_one();
        }
        // if (!results_group.cur_img.empty())
        // {
        //     track.sort(results_group.cur_img, results_group.dets); // 会更新 dets.results
        //     track.showDetection(results_group);

        // }

        // if (!results_group.cur_img.empty())
        // {
        //     if (draw_image_detect(results_group.cur_img, results_group.dets, results_group.cur_frame_id) < 0)
        //     {
        //         printf("save detection results failed!\n");
        //         break;
        //     }
        // }

        frames++;

        // if (frames % 100 == 0)
        // {
        //     gettimeofday(&time, nullptr);
        //     auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
        //     printf("100帧内平均帧率:\t %f fps/s\n", 120.0 / float(currentTime - beforeTime) * 1000.0);
        //     beforeTime = currentTime;
        // }
    }

    // 清空线程池
    while (true)
    {
        DetectResultsGroup results_group;
        if (detectPool.get(results_group) != 0)
            break;

        // if (!results_group.cur_img.empty())
        // // 将检测结果放入跟踪队列
        // {
        //     // cv::Mat img_tmp = results_group.cur_img.clone();
        //     // draw_image_detect(img_tmp, results_group.dets, results_group.cur_frame_id);
        //     std::lock_guard<std::mutex> lock(queueMutex);
        //     trackingQueue.push(results_group);
        //     queueCondVar.notify_one();
        // }

        frames++;
    }

    capture.release();

    // 等待 trackingQueue 清空
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        queueCondVar.wait(lock, []
                          { return trackingQueue.empty(); });
    }

    running = false;
    queueCondVar.notify_all();
    trackingThread.join();

    gettimeofday(&time, nullptr);
    auto endTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    printf("Average:\t %f fps/s\n", float(frames) / float(endTime - startTime) * 1000.0);

    return 0;
}
