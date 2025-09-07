from ultralytics import YOLO

# 加载预训练模型
model = YOLO('D:/yolov8data/yolov8n.pt')  
"""
# 开始训练
model.train(
    data='D:/yolov8data/Chiken2.v2i.yolov8/data.yaml',  # 数据集配置文件路径
    epochs=100,              # 训练轮数
    batch=16,                # 批次大小
    imgsz=640,               # 输入图像大小
    device="cpu"                 # 使用的设备，0 表示第一个 GPU；如果使用 CPU，则设置为 'cpu'
)

"""
model.train(
    # 数据集相关
    data='D:/yolov8data/Chiken2.v2i.yolov8/data.yaml',      
    epochs=100,                    
    batch=21,                        
    imgsz=640,                       
    device="cpu",                      
    workers=8,   
    # 优化后参数                    
    lr0=0.014081425593546793,                        # 初始学习率
    momentum=0.9711200837405246,                  # SGD动量/Adam beta1
    weight_decay=0.00781998446720186,             # 权重衰减
    warmup_epochs=1.0,               # 预热轮数
    optimizer='SGD',                 # 优化器：'SGD', 'Adam', 'AdamW', 'RMSProp'
)