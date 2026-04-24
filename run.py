import time
import torch
import pandas as pd

def benchmark_model_time(model, input_shape, device='cuda', num_warmup=20, num_test=100):
    """
    测量单个模型在特定输入尺寸下的单张图片平均推理时间。
    
    参数:
        model: 需要测试的 PyTorch 模型
        input_shape: 输入张量的形状 (例如: (1, 3, 256, 256) 代表 Batch_size=1, 3通道, 256x256)
        device: 'cuda' 或 'cpu'
        num_warmup: 预热次数 (不计入总时间)
        num_test: 正式测试次数 (用于求平均值)
    """
    model = model.to(device)
    model.eval() # 切换到推理模式，关闭 Dropout 等
    
    # 生成随机的 Dummy 数据模拟图片输入
    dummy_input = torch.randn(input_shape).to(device)
    
    with torch.no_grad(): # 不计算梯度，节省显存和时间
        # 1. 预热阶段 (Warm-up)
        # 消除 GPU 初始化和模型首次加载到显存的 overhead
        for _ in range(num_warmup):
            _ = model(dummy_input)
            
        if device == 'cuda':
            torch.cuda.synchronize() # 等待预热任务在 GPU 上完全结束
            
        # 2. 正式测速阶段
        start_time = time.perf_counter() # 使用高精度计时器
        
        for _ in range(num_test):
            _ = model(dummy_input)
            
        if device == 'cuda':
            torch.cuda.synchronize() # 确保所有 GPU 运算完成再停止计时
            
        end_time = time.perf_counter()
        
    # 计算单张图片的平均处理时间 (秒)
    total_time = end_time - start_time
    avg_time_per_image = total_time / num_test
    
    return avg_time_per_image

def main():
    # 检查是否有 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"正在使用设备: {device}")
    
    # 定义你要测试的分辨率 (以原图表格为准)
    resolutions = [256, 512, 768, 1024]
    
    # 模拟你的模型字典 (这里用简单的卷积层代替，实际应用中替换为你自己的网络实例)
    # 例如: 'Ours': MyUnderwaterModel(), 'C3HLM': C3HLM_Model(), ...
    models_to_test = {
        'C3HLM': torch.nn.Conv2d(3, 3, kernel_size=3, padding=1),
        'MSPE': torch.nn.Conv2d(3, 3, kernel_size=5, padding=2),
        'Ours (我们的)': torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 3, 3, padding=1)
        )
    }
    
    # 用于保存最终结果的字典
    results = {res: [] for res in resolutions}
    results['Method'] = []
    
    print("-" * 50)
    print("开始运行时间基准测试...")
    
    # 遍历每个模型
    for model_name, model in models_to_test.items():
        print(f"\n测试模型: {model_name}")
        results['Method'].append(model_name)
        
        # 遍历每个分辨率
        for res in resolutions:
            input_shape = (1, 3, res, res) # Batch Size 为 1
            
            # 调用测速函数
            avg_time = benchmark_model_time(model, input_shape, device=device)
            
            # 保留三位小数，与你的截图表格保持一致
            results[res].append(round(avg_time, 3))
            print(f"  分辨率 {res}x{res}: {avg_time:.3f} 秒/张")
            
    # 将结果转换为 Pandas DataFrame，方便查看和导出
    # 调整列的顺序，让 Method 在第一列
    df_results = pd.DataFrame(results)
    cols = ['Method'] + resolutions
    df_results = df_results[cols]
    
    print("\n" + "=" * 50)
    print("最终测试结果概览 (单位: 秒/张):")
    print(df_results.to_string(index=False))
    
    # 导出为 CSV，方便直接复制到 Word 或转换为 LaTeX 表格
    df_results.to_csv("running_time_results.csv", index=False)
    print("\n结果已保存至 'running_time_results.csv'")

if __name__ == '__main__':
    main()
