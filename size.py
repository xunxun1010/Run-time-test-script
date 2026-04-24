import os
import cv2
import glob

def resize_dataset():
    # 设置输入和输出的基础路径
    input_dir = r'F:\paper\data_uiedata\uw-UIEB\raw-890'
    output_base_dir = r'F:\paper\data_uiedata\uw-UIEB\resized_datasets'
    
    # 定义需要调整的目标分辨率
    resolutions = [256, 512, 768, 1024]
    
    # 支持的图片格式 (可以根据需要添加 .bmp, .png 等)
    image_extensions = ['*.jpg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        
    if not image_paths:
        print(f"在 {input_dir} 中没有找到图片，请检查路径！")
        return

    print(f"共找到 {len(image_paths)} 张图片，开始处理...")

    # 为每个分辨率创建一个输出文件夹
    for res in resolutions:
        res_dir = os.path.join(output_base_dir, f'size_{res}x{res}')
        os.makedirs(res_dir, exist_ok=True)
        
        print(f"\n正在生成 {res}x{res} 分辨率的图片...")
        
        # 遍历并处理每张图片
        for img_path in image_paths:
            # 获取图片文件名
            img_name = os.path.basename(img_path)
            
            # 读取图片
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取图片: {img_name}")
                continue
            
            # 调整分辨率
            # 注意：如果原图不是正方形，这里会强制拉伸。如果你需要裁剪(Crop)或填充(Pad)，需要修改这里
            resized_img = cv2.resize(img, (res, res), interpolation=cv2.INTER_CUBIC)
            
            # 保存图片
            save_path = os.path.join(res_dir, img_name)
            cv2.imwrite(save_path, resized_img)
            
    print("\n所有图片处理完毕！")

if __name__ == '__main__':
    resize_dataset()