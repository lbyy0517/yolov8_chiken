import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

class Particle:
    """粒子类 - 代表一组超参数"""
    def __init__(self, bounds):
        self.bounds = bounds
        self.position = np.array([
            np.random.uniform(low, high) for low, high in bounds
        ])
        self.velocity = np.zeros(len(bounds))
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.fitness = float('inf')

class PSO:
    """粒子群优化算法"""
    def __init__(self, objective_function, bounds, num_particles=20, max_iter=50, 
                 w=0.7, c1=1.4, c2=1.4):
        self.objective_function = objective_function
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w  # 惯性权重
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 社会学习因子
        
        # 初始化粒子群
        self.particles = [Particle(bounds) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.fitness_history = []

    def optimize(self):
        """执行PSO优化"""
        for iteration in range(self.max_iter):
            for particle in self.particles:
                # 评估当前粒子
                particle.fitness = self.objective_function(particle.position)
                
                # 更新个体最优
                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()
                
                # 更新全局最优
                if particle.fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()
            
            # 更新粒子速度和位置
            for particle in self.particles:
                r1, r2 = np.random.rand(), np.random.rand()
                
                # 更新速度
                cognitive = self.c1 * r1 * (particle.best_position - particle.position)
                social = self.c2 * r2 * (self.global_best_position - particle.position)
                particle.velocity = self.w * particle.velocity + cognitive + social
                
                # 更新位置
                particle.position += particle.velocity
                
                # 边界处理
                for i, (low, high) in enumerate(self.bounds):
                    if particle.position[i] < low:
                        particle.position[i] = low
                    elif particle.position[i] > high:
                        particle.position[i] = high
            
            self.fitness_history.append(self.global_best_fitness)
            print(f"迭代 {iteration+1}/{self.max_iter}, 最佳适应度: {self.global_best_fitness:.4f}")
        
        return self.global_best_position, self.global_best_fitness

class SimpleNN(nn.Module):
    """简单的神经网络模型"""
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2, activation='relu'):
        super(SimpleNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class HyperparameterOptimizer:
    """超参数优化器"""
    def __init__(self, X_train, X_val, y_train, y_val):
        self.X_train = torch.FloatTensor(X_train)
        self.X_val = torch.FloatTensor(X_val)
        self.y_train = torch.LongTensor(y_train)
        self.y_val = torch.LongTensor(y_val)
        
        # 定义超参数搜索空间
        self.bounds = [
            (1e-5, 1e-1),    # 学习率 (log scale)
            (16, 256),       # 批次大小
            (0, 2.99),       # 优化器类型 (0=Adam, 1=SGD, 2=RMSprop)
            (1e-6, 1e-2),    # 权重衰减
            (0.0, 0.5),      # Dropout率
            (1, 4),          # 隐藏层数量
            (32, 512),       # 每层神经元数量
            (0, 2.99),       # 激活函数 (0=relu, 1=tanh, 2=sigmoid)
        ]
        
        self.param_names = [
            'learning_rate', 'batch_size', 'optimizer', 'weight_decay', 
            'dropout_rate', 'num_layers', 'hidden_size', 'activation'
        ]
    
    def decode_parameters(self, position):
        """将粒子位置解码为超参数"""
        params = {}
        
        # 学习率 (对数尺度)
        params['learning_rate'] = 10 ** position[0]
        
        # 批次大小 (整数)
        params['batch_size'] = int(position[1])
        
        # 优化器类型
        optimizer_idx = int(position[2])
        optimizer_map = {0: 'adam', 1: 'sgd', 2: 'rmsprop'}
        params['optimizer'] = optimizer_map[optimizer_idx]
        
        # 权重衰减 (对数尺度)
        params['weight_decay'] = 10 ** position[3]
        
        # Dropout率
        params['dropout_rate'] = position[4]
        
        # 网络结构
        params['num_layers'] = int(position[5])
        params['hidden_size'] = int(position[6])
        
        # 激活函数
        activation_idx = int(position[7])
        activation_map = {0: 'relu', 1: 'tanh', 2: 'sigmoid'}
        params['activation'] = activation_map[activation_idx]
        
        return params
    
    def objective_function(self, position):
        """目标函数 - 训练模型并返回验证损失"""
        try:
            params = self.decode_parameters(position)
            
            # 创建模型
            input_dim = self.X_train.shape[1]
            hidden_dims = [params['hidden_size']] * params['num_layers']
            output_dim = len(torch.unique(self.y_train))
            
            model = SimpleNN(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                dropout_rate=params['dropout_rate'],
                activation=params['activation']
            )
            
            # 创建优化器
            if params['optimizer'] == 'adam':
                optimizer = optim.Adam(model.parameters(), 
                                     lr=params['learning_rate'],
                                     weight_decay=params['weight_decay'])
            elif params['optimizer'] == 'sgd':
                optimizer = optim.SGD(model.parameters(), 
                                    lr=params['learning_rate'],
                                    weight_decay=params['weight_decay'],
                                    momentum=0.9)
            else:  # rmsprop
                optimizer = optim.RMSprop(model.parameters(), 
                                        lr=params['learning_rate'],
                                        weight_decay=params['weight_decay'])
            
            # 创建数据加载器
            train_dataset = TensorDataset(self.X_train, self.y_train)
            train_loader = DataLoader(train_dataset, 
                                    batch_size=params['batch_size'], 
                                    shuffle=True)
            
            # 训练模型
            criterion = nn.CrossEntropyLoss()
            model.train()
            
            epochs = 20  # 为了速度，使用较少的epoch
            for epoch in range(epochs):
                total_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_outputs = model(self.X_val)
                val_loss = criterion(val_outputs, self.y_val).item()
                
                # 计算准确率
                _, predicted = torch.max(val_outputs, 1)
                accuracy = (predicted == self.y_val).float().mean().item()
            
            # 返回验证损失（我们要最小化它）
            # 可以结合损失和准确率
            fitness = val_loss + (1 - accuracy)
            
            return fitness
            
        except Exception as e:
            # 如果出现错误，返回一个很大的惩罚值
            print(f"错误: {e}")
            return 1000.0
    
    def optimize_hyperparameters(self, num_particles=15, max_iter=30):
        """执行超参数优化"""
        print("开始粒子群优化算法优化超参数...")
        print(f"搜索空间: {len(self.bounds)} 个超参数")
        print(f"粒子数量: {num_particles}")
        print(f"最大迭代次数: {max_iter}")
        print("-" * 50)
        
        # 创建PSO优化器
        pso = PSO(
            objective_function=self.objective_function,
            bounds=self.bounds,
            num_particles=num_particles,
            max_iter=max_iter
        )
        
        # 执行优化
        best_position, best_fitness = pso.optimize()
        best_params = self.decode_parameters(best_position)
        
        print("\n" + "="*50)
        print("优化完成！")
        print(f"最佳适应度: {best_fitness:.4f}")
        print("\n最优超参数:")
        for name, value in best_params.items():
            if name in ['learning_rate', 'weight_decay']:
                print(f"  {name}: {value:.2e}")
            elif name in ['batch_size', 'num_layers', 'hidden_size']:
                print(f"  {name}: {int(value)}")
            else:
                print(f"  {name}: {value}")
        
        # 绘制收敛曲线
        plt.figure(figsize=(10, 6))
        plt.plot(pso.fitness_history)
        plt.title('PSO 优化收敛曲线')
        plt.xlabel('迭代次数')
        plt.ylabel('最佳适应度')
        plt.grid(True)
        plt.show()
        
        return best_params, best_fitness

# 使用示例
def main():
    """主函数 - 演示如何使用PSO优化超参数"""
    
    # 生成示例数据
    print("生成示例数据...")
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    # 数据预处理
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    print(f"特征数量: {X_train.shape[1]}")
    print(f"类别数量: {len(np.unique(y))}")
    
    # 创建优化器
    optimizer = HyperparameterOptimizer(X_train, X_val, y_train, y_val)
    
    # 执行优化
    best_params, best_fitness = optimizer.optimize_hyperparameters(
        num_particles=12,  # 可以根据计算资源调整
        max_iter=20       # 可以根据时间要求调整
    )
    
    return best_params, best_fitness

if __name__ == "__main__":
    # 设置随机种子确保结果可重现
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    # 运行优化
    best_params, best_fitness = main()