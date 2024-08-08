# RK3588+deepsort 目标检测与跟踪

## c++版本使用说明
* "data"文件夹存放测试数据
* "model"文件夹存放模型（.rknn）
* 使用说明
```
mkdir build && cd build
cmake ..
make -j8
./rknn_yolov5sort_demo <rknn模型> <reid模型> <视频>
```

## python版本使用说明
python版本的rknn-deepsort好像资料比较少，也没有找到太多依据（借鉴），自己按照pytorch版本改的.(具体参数设置参考main.py的args说明)
```
python main.py
```


## 程序简介
* step-1: 使用线程池来实现目标检测的NPU多线程推理
* step-2: 将推理结果传给deepsort类实现多目标跟踪
目标检测所用模型为[官方](https://github.com/airockchip/rknn_model_zoo)提供onnx自己convert的，若修改为其他模型可能需要修改postprocess部分。
deepsort模型为[大神](https://github.com/leafqycc/rknn-cpp-Multithreading)提供的pt自己转换的onnx、rknn，转换代码[参考]().
python版本的yolov8+deepsort参考了[yolov8](https://git.bwbot.org/publish/rknn3588-yolov8),
[deepsort](https://github.com/ZQPei/deep_sort_pytorch)
c++版本的yolov5+deepsort参考了[yolov5](https://github.com/Zhou-sx/yolov5_Deepsort_rknn),
[deepsort](https://github.com/leafqycc/rknn-cpp-Multithreading)
c++版本可以运行，但感觉线程处理部分有待优化，后续优化会再补充！！！
python版本比较完善了。

## 遇到的问题
1. python版本的yolov8+deepsort帧频为10帧（切换到性能模式可提升至15帧）
目标检测采用线程池技术共两个线程，reid推理没有另外开启线程,在主线程运行
单独运行yolov8帧频变化不大，reid推理对整体处理速度影响不大
2. c++版本的yolov5+deepsort在性能模式下运行帧频为：
15帧：reid单独开一个子线程
18帧：reid合并到主线程
70帧：单独运行目标检测模型

在普通模式下运行帧频为：
16帧：reid单独开一个子线程
16帧：reid合并到主线程
56帧：单独运行目标检测模型

注：目标检测均采用线程池技术共两个线程。

* 下一步计划：
1. 之前只关注了跟踪效果问题，使用yoloV8nas/yolov5nas可显著提高跟踪效果，但是模型太大了不适合实时检测；只能从deepsort跟踪入手，修改特征提取模型可提高reid精度，但是在cascade部分存在马氏距离匹配，当目标消失时间过程，卡尔曼位置预测不再准备，马氏距离会将已经特征匹配上的track和detect取消，所以应该从非线性卡尔曼滤波入手。
2. yolov5s和yolov8s感觉效果差不多，可能只是yolov5材料多可借鉴的多一些。
3. 下一步需要做的事情包括：
    - 补全yolov5的python版本、yolov8的c++版本（三颗星）；
    - 搜集无人机视角数据集重新训练模型（三颗星）；
    - 搜集reid模型（三颗星）；
    - 思考非线性卡尔曼滤波（两颗星）；


## Acknowledgements
 在此特别鸣谢各位大佬，拼接侠拼拼凑凑
* https://github.com/abewley/sort.git
* https://github.com/YunYang1994/OpenWork/tree/master/sort
* https://github.com/mcximing/sort-cpp.git
* https://github.com/Zhou-sx/yolov5_Deepsort_rknn
* https://github.com/leafqycc/rknn-cpp-Multithreading
* https://git.bwbot.org/publish/rknn3588-yolov8
* https://github.com/ZQPei/deep_sort_pytorch
