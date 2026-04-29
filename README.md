# Run-time-test-script
run.py 运行时间测试  
时间测试没有计算imread（读取）和imwrite（写入）带来的硬盘I/O开销,测试采用虚拟图像，只计算了核心算法代码所消耗的时间  
size.py 图片分辨率重整  
timing_summary.csv ours模型时间推理测试  
主要测试了UIEBR890图像，分别对256*256，512*512，768*768的图像进行了图像测速  
