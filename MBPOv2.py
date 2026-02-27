import gymnasium as gym  # 导入gymnasium库，用于创建和管理强化学习环境
import itertools  # 导入itertools库，提供无限计数器等迭代工具
import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch神经网络函数式接口（如ReLU、softplus等）
from torch.distributions import Normal  # 从PyTorch分布模块中导入正态分布类
import numpy as np  # 导入NumPy数值计算库
import random  # 导入Python标准随机数模块
import matplotlib.pyplot as plt  # 导入Matplotlib绑图库
from operator import itemgetter  # 导入itemgetter，用于根据索引快速获取元素
from torch.utils.tensorboard import SummaryWriter  # 导入TensorBoard日志记录器
from datetime import datetime  # 导入datetime模块，用于获取当前时间戳

# 自动选择计算设备：如果有CUDA GPU则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_SIG_MAX = 2  # 策略网络中对数标准差的上界，防止标准差过大
LOG_SIG_MIN = -20  # 策略网络中对数标准差的下界，防止标准差过小（趋近于0）
epsilon = 1e-6  # 极小常数，用于数值稳定性，防止除零或对数运算出错


# ======================== SAC 网络 (对齐 sac/model.py) ========================

def weights_init_(m):
    """自定义权重初始化函数，对所有线性层应用Xavier均匀初始化"""
    if isinstance(m, nn.Linear):  # 如果当前模块是线性层
        torch.nn.init.xavier_uniform_(m.weight, gain=1)  # 使用Xavier均匀初始化权重，gain=1
        torch.nn.init.constant_(m.bias, 0)  # 将偏置初始化为0


class QNetwork(nn.Module):
    """双Q网络（Twin Q-Network），SAC算法中用于估计状态-动作值函数Q(s,a)"""
    def __init__(self, num_inputs, num_actions, hidden_dim):
        """
        初始化双Q网络
        参数:
            num_inputs: 状态空间维度
            num_actions: 动作空间维度
            hidden_dim: 隐藏层神经元数量
        """
        super(QNetwork, self).__init__()  # 调用父类nn.Module的构造函数

        # ---- 第一个Q网络 (Q1) ----
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)  # Q1输入层：状态+动作 -> 隐藏层
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)  # Q1隐藏层：隐藏层 -> 隐藏层
        self.linear3 = nn.Linear(hidden_dim, 1)  # Q1输出层：隐藏层 -> Q值（标量）

        # ---- 第二个Q网络 (Q2) ----
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)  # Q2输入层：状态+动作 -> 隐藏层
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)  # Q2隐藏层：隐藏层 -> 隐藏层
        self.linear6 = nn.Linear(hidden_dim, 1)  # Q2输出层：隐藏层 -> Q值（标量）

        self.apply(weights_init_)  # 对所有子模块应用Xavier权重初始化

    def forward(self, state, action):
        """
        前向传播，同时计算两个Q网络的输出
        参数:
            state: 状态张量
            action: 动作张量
        返回:
            x1, x2: 两个Q网络分别输出的Q值
        """
        xu = torch.cat([state, action], 1)  # 将状态和动作在维度1上拼接，作为网络输入

        # Q1前向传播
        x1 = F.relu(self.linear1(xu))  # 第一层线性变换后接ReLU激活函数
        x1 = F.relu(self.linear2(x1))  # 第二层线性变换后接ReLU激活函数
        x1 = self.linear3(x1)  # 输出层，不使用激活函数，直接输出Q值

        # Q2前向传播
        x2 = F.relu(self.linear4(xu))  # 第一层线性变换后接ReLU激活函数
        x2 = F.relu(self.linear5(x2))  # 第二层线性变换后接ReLU激活函数
        x2 = self.linear6(x2)  # 输出层，不使用激活函数，直接输出Q值
        return x1, x2  # 返回两个Q网络的Q值


class GaussianPolicy(nn.Module):
    """高斯策略网络，SAC算法中用于输出连续动作的随机策略"""
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        """
        初始化高斯策略网络
        参数:
            num_inputs: 状态空间维度
            num_actions: 动作空间维度
            hidden_dim: 隐藏层神经元数量
            action_space: 动作空间对象，用于计算动作的缩放和偏移
        """
        super(GaussianPolicy, self).__init__()  # 调用父类构造函数

        self.linear1 = nn.Linear(num_inputs, hidden_dim)  # 输入层：状态 -> 隐藏层
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层：隐藏层 -> 隐藏层

        self.mean_linear = nn.Linear(hidden_dim, num_actions)  # 均值输出层：隐藏层 -> 动作均值
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)  # 对数标准差输出层：隐藏层 -> 动作对数标准差

        self.apply(weights_init_)  # 对所有子模块应用Xavier权重初始化

        if action_space is None:  # 如果没有提供动作空间
            self.action_scale = torch.tensor(1.)  # 动作缩放因子设为1（不缩放）
            self.action_bias = torch.tensor(0.)  # 动作偏移量设为0（不偏移）
        else:  # 如果提供了动作空间，根据动作范围计算缩放和偏移
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)  # 缩放因子 = (上界-下界)/2
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)  # 偏移量 = (上界+下界)/2

    def forward(self, state):
        """
        前向传播，输出动作分布的均值和对数标准差
        参数:
            state: 状态张量
        返回:
            mean: 动作分布的均值
            log_std: 动作分布的对数标准差（经过裁剪）
        """
        x = F.relu(self.linear1(state))  # 第一层线性变换后接ReLU激活函数
        x = F.relu(self.linear2(x))  # 第二层线性变换后接ReLU激活函数
        mean = self.mean_linear(x)  # 计算动作分布的均值
        log_std = self.log_std_linear(x)  # 计算动作分布的对数标准差
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)  # 将对数标准差裁剪到[LOG_SIG_MIN, LOG_SIG_MAX]范围内
        return mean, log_std  # 返回均值和对数标准差

    def sample(self, state):
        """
        从策略中采样动作（使用重参数化技巧）
        参数:
            state: 状态张量
        返回:
            action: 采样得到的动作（经过tanh压缩和缩放）
            log_prob: 动作的对数概率
            mean: 确定性动作（均值经过tanh压缩和缩放）
        """
        mean, log_std = self.forward(state)  # 通过前向传播获取均值和对数标准差
        std = log_std.exp()  # 将对数标准差转换为标准差：std = exp(log_std)
        normal = Normal(mean, std)  # 创建正态分布对象 N(mean, std)
        x_t = normal.rsample()  # 使用重参数化技巧采样（允许梯度回传）：x_t = mean + std * epsilon
        y_t = torch.tanh(x_t)  # 对采样值应用tanh压缩到(-1, 1)范围
        action = y_t * self.action_scale + self.action_bias  # 将(-1,1)范围的动作缩放到实际动作空间
        log_prob = normal.log_prob(x_t)  # 计算原始采样值x_t在正态分布下的对数概率
        # 应用tanh变换的雅可比矩阵修正，得到变换后动作的对数概率
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)  # 将各维度的对数概率求和（联合概率的对数）
        mean = torch.tanh(mean) * self.action_scale + self.action_bias  # 计算确定性动作（均值经过tanh和缩放）
        return action, log_prob, mean  # 返回采样动作、对数概率、确定性动作

    def to(self, device):
        """
        将模型及其缓冲区移动到指定设备（GPU/CPU）
        参数:
            device: 目标设备
        返回:
            移动后的模型对象
        """
        self.action_scale = self.action_scale.to(device)  # 将动作缩放因子移到目标设备
        self.action_bias = self.action_bias.to(device)  # 将动作偏移量移到目标设备
        return super(GaussianPolicy, self).to(device)  # 调用父类to方法移动其余参数


# ======================== SAC utils (对齐 sac/utils.py) ========================

def soft_update(target, source, tau):
    """
    软更新（Polyak平均）：target = (1-tau)*target + tau*source
    参数:
        target: 目标网络
        source: 源网络
        tau: 软更新系数，通常很小（如0.005）
    """
    for target_param, param in zip(target.parameters(), source.parameters()):  # 逐参数遍历目标网络和源网络
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)  # 用加权平均更新目标参数


def hard_update(target, source):
    """
    硬更新：直接将源网络参数复制到目标网络
    参数:
        target: 目标网络
        source: 源网络
    """
    for target_param, param in zip(target.parameters(), source.parameters()):  # 逐参数遍历
        target_param.data.copy_(param.data)  # 直接复制源网络参数到目标网络


# ======================== SAC Agent (对齐 sac/sac.py) ========================

class SAC(object):
    """SAC（Soft Actor-Critic）算法实现，基于最大熵强化学习框架"""
    def __init__(self, num_inputs, action_space, args):
        """
        初始化SAC智能体
        参数:
            num_inputs: 状态空间维度
            action_space: 动作空间对象
            args: 超参数配置对象
        """
        self.gamma = args.gamma  # 折扣因子，用于计算未来奖励的折扣值
        self.tau = args.tau  # 软更新系数，用于目标网络的Polyak更新
        self.alpha = args.alpha  # 熵正则化系数（温度参数），控制探索与利用的平衡

        self.target_update_interval = args.target_update_interval  # 目标网络更新间隔（每多少步更新一次）
        self.automatic_entropy_tuning = args.automatic_entropy_tuning  # 是否自动调整熵系数alpha

        self.device = device  # 计算设备（GPU或CPU）

        # 创建Critic网络（双Q网络）并移至计算设备
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.lr)  # Critic的Adam优化器

        # 创建目标Critic网络并硬拷贝参数
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)  # 用硬更新将critic参数完全复制到critic_target

        if self.automatic_entropy_tuning:  # 如果启用自动熵调整
            # 目标熵设为 -dim(A)，即动作空间维度的负数
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            # 初始化可学习的对数alpha参数（初始值为0，即alpha=1）
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.lr)  # alpha的Adam优化器

        # 创建策略网络（高斯策略）并移至计算设备
        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=args.lr)  # 策略网络的Adam优化器

    def select_action(self, state, eval=False):
        """
        根据当前状态选择动作
        参数:
            state: 当前状态（numpy数组）
            eval: 是否为评估模式，True则使用确定性策略（均值），False则使用随机采样
        返回:
            action: 选择的动作（numpy数组）
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)  # 将状态转为张量并添加batch维度
        if not eval:  # 训练模式：使用随机采样的动作（鼓励探索）
            action, _, _ = self.policy.sample(state)  # 从策略分布中随机采样
        else:  # 评估模式：使用确定性动作（均值）
            _, _, action = self.policy.sample(state)  # 使用策略的确定性输出（均值）
        return action.detach().cpu().numpy()[0]  # 将动作从GPU移到CPU，转为numpy数组，去掉batch维度

    def update_parameters(self, memory, batch_size, updates):
        """
        更新SAC的所有网络参数（Critic、Policy、Alpha）
        参数:
            memory: 经验数据元组 (state, action, reward, next_state, mask)
            batch_size: 批量大小
            updates: 当前更新步数，用于控制目标网络更新频率
        返回:
            info_dict: 包含所有SAC训练指标的字典
        """
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        qf1, qf2 = self.critic(state_batch, action_batch)

        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), float('inf'))
        self.policy_optim.step()

        self.critic_optim.zero_grad()
        (qf1_loss + qf2_loss).backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float('inf'))
        self.critic_optim.step()

        info_dict = {
            'qf1_loss': qf1_loss.item(),
            'qf2_loss': qf2_loss.item(),
            'policy_loss': policy_loss.item(),
            'qf1_mean': qf1.mean().item(),
            'qf2_mean': qf2.mean().item(),
            'qf1_std': qf1.std().item(),
            'qf2_std': qf2.std().item(),
            'target_q_mean': next_q_value.mean().item(),
            'target_q_std': next_q_value.std().item(),
            'log_pi_mean': log_pi.mean().item(),
            'log_pi_std': log_pi.std().item(),
            'policy_mean_abs': pi.abs().mean().item(),
            'policy_std': pi.std().item(),
            'reward_mean': reward_batch.mean().item(),
            'reward_std': reward_batch.std().item(),
            'policy_grad_norm': policy_grad_norm.item(),
            'critic_grad_norm': critic_grad_norm.item(),
        }

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

            info_dict['alpha_loss'] = alpha_loss.item()
            info_dict['alpha'] = self.alpha.item()
            info_dict['target_entropy'] = self.target_entropy
        else:
            info_dict['alpha'] = self.alpha

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return info_dict


# ======================== ReplayMemory (对齐 sac/replay_memory.py) ========================

class ReplayMemory:
    """经验回放缓冲区，用于存储和采样交互经验"""
    def __init__(self, capacity):
        """
        初始化回放缓冲区
        参数:
            capacity: 缓冲区最大容量
        """
        self.capacity = capacity  # 缓冲区最大容量
        self.buffer = []  # 用列表存储经验数据
        self.position = 0  # 当前写入位置的指针（用于环形覆盖）

    def push(self, state, action, reward, next_state, done):
        """
        向缓冲区中添加单条经验
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
        """
        if len(self.buffer) < self.capacity:  # 如果缓冲区未满
            self.buffer.append(None)  # 先扩展一个空位
        self.buffer[self.position] = (state, action, reward, next_state, done)  # 将经验存储到当前位置
        self.position = (self.position + 1) % self.capacity  # 移动指针，到达末尾时回到开头（环形缓冲区）

    def push_batch(self, batch):
        """
        批量添加经验数据到缓冲区
        参数:
            batch: 经验数据列表
        """
        if len(self.buffer) < self.capacity:  # 如果缓冲区未满
            append_len = min(self.capacity - len(self.buffer), len(batch))  # 计算需要新增的空位数量
            self.buffer.extend([None] * append_len)  # 扩展缓冲区

        if self.position + len(batch) < self.capacity:  # 如果批量数据不会超过缓冲区末尾
            self.buffer[self.position: self.position + len(batch)] = batch  # 直接写入
            self.position += len(batch)  # 移动指针
        else:  # 如果批量数据会越过缓冲区末尾（需要环形写入）
            self.buffer[self.position: len(self.buffer)] = batch[:len(self.buffer) - self.position]  # 写入到末尾
            self.buffer[:len(batch) - len(self.buffer) + self.position] = batch[len(self.buffer) - self.position:]  # 剩余部分从头写入
            self.position = len(batch) - len(self.buffer) + self.position  # 更新指针位置

    def sample(self, batch_size):
        """
        从缓冲区中随机采样一批经验（不重复采样）
        参数:
            batch_size: 采样数量
        返回:
            state, action, reward, next_state, done: 各数据的numpy数组
        """
        if batch_size > len(self.buffer):  # 如果采样数量大于缓冲区大小
            batch_size = len(self.buffer)  # 将采样数量限制为缓冲区大小
        batch = random.sample(self.buffer, int(batch_size))  # 从缓冲区中随机采样batch_size条数据（不放回）
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # 将采样数据按字段拆分并堆叠为numpy数组
        return state, action, reward, next_state, done  # 返回分离后的各字段数组

    def sample_all_batch(self, batch_size):
        """
        从缓冲区中随机采样一批经验（可重复采样）
        参数:
            batch_size: 采样数量
        返回:
            state, action, reward, next_state, done: 各数据的numpy数组
        """
        idxes = np.random.randint(0, len(self.buffer), batch_size)  # 随机生成batch_size个索引（有放回，可能重复）
        batch = list(itemgetter(*idxes)(self.buffer))  # 根据索引从缓冲区获取对应的经验数据
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # 按字段拆分并堆叠为numpy数组
        return state, action, reward, next_state, done  # 返回分离后的各字段数组

    def return_all(self):
        """返回缓冲区中所有数据"""
        return self.buffer

    def __len__(self):
        """返回缓冲区中当前存储的经验数量"""
        return len(self.buffer)


# ======================== Ensemble 环境模型 (对齐 model.py) ========================

class StandardScaler(object):
    """标准化器，对数据进行零均值单位方差的标准化处理"""
    def __init__(self):
        pass  # 构造函数为空，均值和方差在fit时计算

    def fit(self, data):
        """
        根据数据计算均值和标准差
        参数:
            data: 用于计算统计量的数据（numpy数组）
        """
        self.mu = np.mean(data, axis=0, keepdims=True)  # 按列计算均值，保持维度
        self.std = np.std(data, axis=0, keepdims=True)  # 按列计算标准差，保持维度
        self.std[self.std < 1e-12] = 1.0  # 将过小的标准差设为1，避免除零

    def transform(self, data):
        """
        对数据进行标准化（零均值单位方差）
        参数:
            data: 待标准化的数据
        返回:
            标准化后的数据
        """
        return (data - self.mu) / self.std  # 标准化公式：(x - μ) / σ

    def inverse_transform(self, data):
        """
        将标准化后的数据还原为原始尺度
        参数:
            data: 标准化后的数据
        返回:
            还原后的数据
        """
        return self.std * data + self.mu  # 逆标准化公式：σ * x + μ


class Swish(nn.Module):
    """Swish激活函数：f(x) = x * sigmoid(x)，比ReLU更平滑"""
    def __init__(self):
        super(Swish, self).__init__()  # 调用父类构造函数

    def forward(self, x):
        """前向传播，计算Swish激活：x * sigmoid(x)"""
        x = x * F.sigmoid(x)  # Swish激活函数：x乘以sigmoid(x)
        return x


def init_weights(m):
    """
    自定义权重初始化函数，使用截断正态分布初始化
    参数:
        m: 神经网络模块
    """
    def truncated_normal_init(t, mean=0.0, std=0.01):
        """
        截断正态分布初始化：将超出±2σ范围的值重新采样
        参数:
            t: 待初始化的张量
            mean: 均值
            std: 标准差
        返回:
            初始化后的张量
        """
        torch.nn.init.normal_(t, mean=mean, std=std)  # 先用标准正态分布初始化
        while True:  # 循环直到所有值都在[mean-2std, mean+2std]范围内
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)  # 找出超出范围的值
            if not torch.sum(cond):  # 如果没有超出范围的值，退出循环
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)  # 对超出范围的值重新采样
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):  # 如果是线性层或集成全连接层
        input_dim = m.in_features  # 获取输入维度
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))  # 使用截断正态分布初始化权重，标准差与输入维度相关
        m.bias.data.fill_(0.0)  # 偏置初始化为0


class EnsembleFC(nn.Module):
    """
    集成全连接层：同时维护多个并行的全连接层
    通过批量矩阵乘法(BMM)高效计算多个模型的前向传播
    """
    __constants__ = ['in_features', 'out_features']  # TorchScript常量声明
    in_features: int  # 输入特征维度
    out_features: int  # 输出特征维度
    ensemble_size: int  # 集成模型数量
    weight: torch.Tensor  # 权重张量

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True) -> None:
        """
        初始化集成全连接层
        参数:
            in_features: 输入特征维度
            out_features: 输出特征维度
            ensemble_size: 集成模型数量
            weight_decay: 权重衰减系数（L2正则化）
            bias: 是否使用偏置
        """
        super(EnsembleFC, self).__init__()  # 调用父类构造函数
        self.in_features = in_features  # 保存输入维度
        self.out_features = out_features  # 保存输出维度
        self.ensemble_size = ensemble_size  # 保存集成模型数量
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))  # 创建权重参数：(集成数, 输入维度, 输出维度)
        self.weight_decay = weight_decay  # 保存权重衰减系数
        if bias:  # 如果使用偏置
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))  # 创建偏置参数：(集成数, 输出维度)
        else:
            self.register_parameter('bias', None)  # 不使用偏置时注册为None
        self.reset_parameters()  # 调用参数重置（此处为空操作）

    def reset_parameters(self) -> None:
        """参数重置（留空，由init_weights函数统一处理初始化）"""
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播：使用批量矩阵乘法并行计算所有集成模型的输出
        参数:
            input: 输入张量，形状为 (ensemble_size, batch_size, in_features)
        返回:
            输出张量，形状为 (ensemble_size, batch_size, out_features)
        """
        w_times_x = torch.bmm(input, self.weight)  # 批量矩阵乘法：(E,B,in) @ (E,in,out) -> (E,B,out)
        return torch.add(w_times_x, self.bias[:, None, :])  # 加上偏置（广播到batch维度）

    def extra_repr(self) -> str:
        """返回模块的额外字符串表示（用于print时显示）"""
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class EnsembleModel(nn.Module):
    """
    集成动力学模型：使用多个神经网络并行预测环境动力学
    输入(state, action)，输出(reward, next_state - state)的均值和方差
    """
    def __init__(self, state_size, action_size, reward_size, ensemble_size, hidden_size=200, learning_rate=1e-3, use_decay=False):
        """
        初始化集成模型
        参数:
            state_size: 状态维度
            action_size: 动作维度
            reward_size: 奖励维度（通常为1）
            ensemble_size: 集成模型数量
            hidden_size: 隐藏层大小
            learning_rate: 学习率
            use_decay: 是否使用权重衰减（L2正则化）
        """
        super(EnsembleModel, self).__init__()  # 调用父类构造函数
        self.hidden_size = hidden_size  # 保存隐藏层大小
        # 四层集成全连接层，每层设置不同的权重衰减系数
        self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size, weight_decay=0.000025)  # 第一层：(state+action) -> hidden
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)  # 第二层：hidden -> hidden
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)  # 第三层：hidden -> hidden
        self.nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)  # 第四层：hidden -> hidden
        self.use_decay = use_decay  # 是否使用权重衰减

        self.output_dim = state_size + reward_size  # 输出维度 = 状态维度 + 奖励维度
        # 输出层：预测均值和对数方差，因此输出维度翻倍
        self.nn5 = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)

        # 对数方差的上界参数（可学习，但此处requires_grad=False表示不通过梯度更新，而是通过损失中的正则项间接调整）
        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim)).float() / 2).to(device), requires_grad=False)
        # 对数方差的下界参数
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim)).float() * 10).to(device), requires_grad=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)  # Adam优化器
        self.apply(init_weights)  # 对所有子模块应用截断正态分布权重初始化
        self.swish = Swish()  # 创建Swish激活函数实例
        self.to(device)  # 将模型移至计算设备

    def forward(self, x, ret_log_var=False):
        """
        前向传播
        参数:
            x: 输入张量 (ensemble_size, batch_size, state_dim+action_dim)
            ret_log_var: 如果True则返回对数方差，否则返回方差
        返回:
            mean: 预测均值
            logvar/var: 对数方差或方差
        """
        nn1_output = self.swish(self.nn1(x))  # 第一层：线性变换 + Swish激活
        nn2_output = self.swish(self.nn2(nn1_output))  # 第二层：线性变换 + Swish激活
        nn3_output = self.swish(self.nn3(nn2_output))  # 第三层：线性变换 + Swish激活
        nn4_output = self.swish(self.nn4(nn3_output))  # 第四层：线性变换 + Swish激活
        nn5_output = self.nn5(nn4_output)  # 输出层：线性变换（不使用激活函数）

        mean = nn5_output[:, :, :self.output_dim]  # 前半部分为预测均值

        # 使用softplus将对数方差限制在[min_logvar, max_logvar]范围内
        logvar = self.max_logvar - F.softplus(self.max_logvar - nn5_output[:, :, self.output_dim:])  # 上界约束
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)  # 下界约束

        if ret_log_var:  # 如果需要返回对数方差
            return mean, logvar
        else:  # 否则返回方差
            return mean, torch.exp(logvar)  # 方差 = exp(对数方差)

    def get_decay_loss(self):
        """
        计算所有集成全连接层的权重衰减损失（L2正则化）
        返回:
            decay_loss: 权重衰减损失值
        """
        decay_loss = 0.  # 初始化衰减损失
        for m in self.children():  # 遍历所有子模块
            if isinstance(m, EnsembleFC):  # 如果是集成全连接层
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.  # 累加 L2正则化项
        return decay_loss

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        """
        计算高斯负对数似然损失
        参数:
            mean: 预测均值 (ensemble_size, batch_size, output_dim)
            logvar: 预测对数方差 (ensemble_size, batch_size, output_dim)
            labels: 真实标签 (ensemble_size, batch_size, output_dim)
            inc_var_loss: 是否包含方差损失项
        返回:
            total_loss: 总损失
            mse_loss: 各模型的MSE损失
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3  # 确保输入是3维张量
        inv_var = torch.exp(-logvar)  # 方差的倒数：1/σ² = exp(-log(σ²))
        if inc_var_loss:  # 如果包含方差损失（完整的高斯NLL）
            # MSE损失加权：(mean - label)² / σ²
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)  # 方差损失：log(σ²)的均值
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)  # 总损失 = 加权MSE + 方差惩罚
        else:  # 不包含方差损失（纯MSE）
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))  # 计算各模型的MSE损失
            total_loss = torch.sum(mse_loss)  # 总损失 = 所有模型MSE之和
        return total_loss, mse_loss  # 返回总损失和各模型的MSE损失

    def train(self, loss):
        """
        执行一步梯度更新
        参数:
            loss: 需要最小化的损失值
        """
        self.optimizer.zero_grad()  # 清除优化器中的梯度
        # 添加对max_logvar和min_logvar的正则化：鼓励max_logvar变小、min_logvar变大，收紧方差范围
        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        if self.use_decay:  # 如果使用权重衰减
            loss += self.get_decay_loss()  # 加上L2权重衰减损失
        loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 更新模型参数


class EnsembleDynamicsModel():
    """
    集成动力学模型管理器
    管理集成模型的训练、验证、精英模型选择和预测
    """
    def __init__(self, network_size, elite_size, state_size, action_size, reward_size=1, hidden_size=200, use_decay=False):
        """
        初始化集成动力学模型
        参数:
            network_size: 集成中的模型总数
            elite_size: 精英模型数量（预测时从中随机选择）
            state_size: 状态维度
            action_size: 动作维度
            reward_size: 奖励维度
            hidden_size: 隐藏层大小
            use_decay: 是否使用权重衰减
        """
        self.network_size = network_size  # 集成模型总数
        self.elite_size = elite_size  # 精英模型数量
        self.model_list = []  # 模型列表（此处未使用，使用ensemble_model代替）
        self.state_size = state_size  # 状态维度
        self.action_size = action_size  # 动作维度
        self.reward_size = reward_size  # 奖励维度
        self.elite_model_idxes = []  # 精英模型的索引列表
        # 创建集成模型实例
        self.ensemble_model = EnsembleModel(state_size, action_size, reward_size, network_size, hidden_size, use_decay=use_decay)
        self.scaler = StandardScaler()  # 创建数据标准化器

    def train(self, inputs, labels, batch_size=256, holdout_ratio=0.2, max_epochs_since_update=5):
        """
        训练集成动力学模型
        参数:
            inputs: 训练输入 (state, action) 拼接
            labels: 训练标签 (reward, delta_state) 拼接
            batch_size: 训练批量大小
            holdout_ratio: 验证集比例
            max_epochs_since_update: 最大允许无改善轮数（早停条件）
        返回:
            model_info: 包含模型训练指标的字典
        """
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}

        num_holdout = int(inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]

        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])

        epoch_train_losses = []

        for epoch in itertools.count():
            train_idx = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self.network_size)])
            epoch_loss_sum = 0.0
            num_batches = 0
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[:, start_pos: start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[idx]).float().to(device)
                train_label = torch.from_numpy(train_labels[idx]).float().to(device)
                mean, logvar = self.ensemble_model(train_input, ret_log_var=True)
                loss, _ = self.ensemble_model.loss(mean, logvar, train_label)
                self.ensemble_model.train(loss)
                epoch_loss_sum += loss.item()
                num_batches += 1

            epoch_train_losses.append(epoch_loss_sum / max(num_batches, 1))

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.ensemble_model(holdout_inputs, ret_log_var=True)
                _, holdout_mse_losses = self.ensemble_model.loss(holdout_mean, holdout_logvar, holdout_labels, inc_var_loss=False)
                holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                sorted_loss_idx = np.argsort(holdout_mse_losses)
                self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()
                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    break

        holdout_var = holdout_logvar.exp()

        model_info = {
            'model_train_epochs': epoch + 1,
            'model_train_loss_final': epoch_train_losses[-1],
            'model_holdout_mse_mean': float(np.mean(holdout_mse_losses)),
            'model_holdout_mse_std': float(np.std(holdout_mse_losses)),
            'model_holdout_mse_min': float(np.min(holdout_mse_losses)),
            'model_holdout_mse_max': float(np.max(holdout_mse_losses)),
            'model_elite_mse_mean': float(np.mean(holdout_mse_losses[self.elite_model_idxes])),
            'model_train_data_size': train_inputs.shape[0],
            'model_holdout_data_size': num_holdout,
            'model_max_logvar_mean': self.ensemble_model.max_logvar.mean().item(),
            'model_min_logvar_mean': self.ensemble_model.min_logvar.mean().item(),
            'model_holdout_var_mean': holdout_var.mean().item(),
            'model_holdout_var_max': holdout_var.max().item(),
        }
        for i in range(self.network_size):
            model_info[f'model_{i}_holdout_mse'] = float(holdout_mse_losses[i])

        return model_info

    def _save_best(self, epoch, holdout_losses):
        """
        保存最佳模型快照并检查早停条件
        参数:
            epoch: 当前训练轮次
            holdout_losses: 各集成模型在验证集上的损失
        返回:
            True表示应停止训练，False表示继续
        """
        updated = False  # 标记是否有任何模型获得了显著改善
        for i in range(len(holdout_losses)):  # 遍历每个集成模型
            current = holdout_losses[i]  # 当前模型的验证损失
            _, best = self._snapshots[i]  # 获取该模型历史最佳损失
            improvement = (best - current) / best  # 计算相对改善率
            if improvement > 0.01:  # 如果改善超过1%
                self._snapshots[i] = (epoch, current)  # 更新最佳快照
                updated = True  # 标记有改善

        if updated:  # 如果有模型获得了改善
            self._epochs_since_update = 0  # 重置无改善计数器
        else:  # 没有任何模型改善
            self._epochs_since_update += 1  # 无改善计数器加1
        if self._epochs_since_update > self._max_epochs_since_update:  # 如果连续无改善轮数超过阈值
            return True  # 触发早停
        else:
            return False  # 继续训练

    def predict(self, inputs, batch_size=1024, factored=True):
        """
        使用集成模型进行预测
        参数:
            inputs: 输入数据 (state, action) 拼接
            batch_size: 预测时的批量大小
            factored: 如果True返回各模型独立的均值和方差，否则返回混合后的结果
        返回:
            ensemble_mean: 各模型预测均值 (ensemble_size, N, output_dim)
            ensemble_var: 各模型预测方差 (ensemble_size, N, output_dim)
        """
        inputs = self.scaler.transform(inputs)  # 对输入进行标准化
        ensemble_mean, ensemble_var = [], []  # 初始化结果列表
        for i in range(0, inputs.shape[0], batch_size):  # 按batch_size分批处理
            input = torch.from_numpy(inputs[i:min(i + batch_size, inputs.shape[0])]).float().to(device)  # 取一个batch并转为张量
            # 复制network_size份并行预测
            b_mean, b_var = self.ensemble_model(input[None, :, :].repeat([self.network_size, 1, 1]), ret_log_var=False)
            ensemble_mean.append(b_mean.detach().cpu().numpy())  # 保存预测均值
            ensemble_var.append(b_var.detach().cpu().numpy())  # 保存预测方差
        ensemble_mean = np.hstack(ensemble_mean)  # 在batch维度上拼接所有均值
        ensemble_var = np.hstack(ensemble_var)  # 在batch维度上拼接所有方差

        if factored:  # 如果返回分解后的结果（各模型独立）
            return ensemble_mean, ensemble_var
        else:  # 否则返回混合结果（未实现）
            assert False, "Need to transform to numpy"  # 断言错误，此分支未完整实现
            mean = torch.mean(ensemble_mean, dim=0)  # 混合均值：所有模型均值的平均
            # 混合方差：各模型方差的均值 + 各模型均值偏差的方差
            var = torch.mean(ensemble_var, dim=0) + torch.mean(torch.square(ensemble_mean - mean[None, :, :]), dim=0)
            return mean, var


# ======================== PredictEnv (对齐 predict_env.py) ========================

class PredictEnv:
    """预测环境：封装集成动力学模型，提供类似gym环境的step接口"""
    def __init__(self, model, env_name, model_type='pytorch'):
        """
        初始化预测环境
        参数:
            model: 集成动力学模型对象
            env_name: 环境名称（用于确定终止条件）
            model_type: 模型类型（默认pytorch）
        """
        self.model = model  # 保存动力学模型
        self.env_name = env_name  # 保存环境名称
        self.model_type = model_type  # 保存模型类型

    def _termination_fn(self, env_name, obs, act, next_obs):
        """
        根据环境类型判断是否终止（done）
        不同环境有不同的终止条件
        参数:
            env_name: 环境名称
            obs: 当前观测
            act: 当前动作
            next_obs: 下一个观测
        返回:
            done: 终止标志数组
        """
        if env_name == "Hopper-v2":  # Hopper环境的终止条件
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2  # 确保输入是2维的
            height = next_obs[:, 0]  # 获取高度（第0维）
            angle = next_obs[:, 1]  # 获取角度（第1维）
            # 未终止条件：所有观测值有限 且 各维度绝对值小于100 且 高度>0.7 且 角度绝对值<0.2
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                       * (height > .7) \
                       * (np.abs(angle) < .2)
            done = ~not_done  # 终止 = 非(未终止)
            done = done[:, None]  # 添加一个维度以匹配其他数据的形状
            return done
        elif env_name == "Walker2d-v2":  # Walker2d环境的终止条件
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2  # 确保输入是2维的
            height = next_obs[:, 0]  # 获取高度
            angle = next_obs[:, 1]  # 获取角度
            # 未终止条件：0.8 < height < 2.0 且 -1.0 < angle < 1.0
            not_done = (height > 0.8) \
                       * (height < 2.0) \
                       * (angle > -1.0) \
                       * (angle < 1.0)
            done = ~not_done  # 终止 = 非(未终止)
            done = done[:, None]  # 添加维度
            return done
        else:  # 其他环境默认不终止
            done = np.zeros((next_obs.shape[0], 1), dtype=bool)  # 全部设为False（不终止）
            return done

    def _get_logprob(self, x, means, variances):
        """
        计算样本在集成模型混合高斯分布下的对数概率
        参数:
            x: 采样值
            means: 各模型的预测均值
            variances: 各模型的预测方差
        返回:
            log_prob: 混合分布的对数概率
            stds: 各模型均值的标准差（衡量模型不确定性）
        """
        k = x.shape[-1]  # 获取输出维度
        # 计算每个模型下的对数概率：log N(x | mean, var)
        log_prob = -1 / 2 * (k * np.log(2 * np.pi) + np.log(variances).sum(-1) + (np.power(x - means, 2) / variances).sum(-1))
        prob = np.exp(log_prob).sum(0)  # 将各模型的概率求和（混合分布）
        log_prob = np.log(prob)  # 取混合概率的对数
        stds = np.std(means, 0).mean(-1)  # 计算各模型均值预测的标准差（反映模型间的分歧/不确定性）
        return log_prob, stds

    def step(self, obs, act, deterministic=False):
        """
        使用动力学模型进行一步模拟（类似gym的step接口）
        参数:
            obs: 当前观测
            act: 当前动作
            deterministic: 是否使用确定性预测（不添加噪声）
        返回:
            next_obs: 预测的下一个观测
            rewards: 预测的奖励
            terminals: 终止标志
            info: 包含均值、标准差、对数概率、模型分歧等信息的字典
        """
        if len(obs.shape) == 1:  # 如果输入是一维的（单个样本）
            obs = obs[None]  # 添加batch维度
            act = act[None]  # 添加batch维度
            return_single = True  # 标记需要返回单个结果
        else:
            return_single = False  # 标记返回批量结果

        inputs = np.concatenate((obs, act), axis=-1)  # 拼接状态和动作作为模型输入
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)  # 用集成模型预测均值和方差
        # 模型预测的是delta_state（状态增量），需要加上当前状态得到next_state
        ensemble_model_means[:, :, self.model.reward_size:] += obs  # 将状态增量加上当前状态
        ensemble_model_stds = np.sqrt(ensemble_model_vars)  # 计算标准差：std = sqrt(var)

        if deterministic:  # 确定性预测
            ensemble_samples = ensemble_model_means  # 直接使用均值作为采样结果
        else:  # 随机预测
            # 均值 + 随机噪声 * 标准差（重参数化采样）
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape  # 获取模型数、批量大小
        model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)  # 为每个样本随机选择一个精英模型
        batch_idxes = np.arange(0, batch_size)  # 创建批量索引

        samples = ensemble_samples[model_idxes, batch_idxes]  # 从选中的模型中获取对应的采样结果
        model_means = ensemble_model_means[model_idxes, batch_idxes]  # 获取选中模型的均值
        model_stds = ensemble_model_stds[model_idxes, batch_idxes]  # 获取选中模型的标准差

        # 计算混合分布的对数概率和模型分歧度
        log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        rewards, next_obs = samples[:, :self.model.reward_size], samples[:, self.model.reward_size:]  # 拆分奖励和下一状态
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)  # 根据环境规则判断是否终止

        batch_size = model_means.shape[0]  # 获取实际的批量大小
        # 拼接返回的均值信息：(奖励均值, 终止标志, 状态均值)
        return_means = np.concatenate((model_means[:, :self.model.reward_size], terminals, model_means[:, self.model.reward_size:]), axis=-1)
        # 拼接返回的标准差信息：(奖励标准差, 0（终止无方差）, 状态标准差)
        return_stds = np.concatenate((model_stds[:, :self.model.reward_size], np.zeros((batch_size, 1)), model_stds[:, self.model.reward_size:]), axis=-1)

        if return_single:  # 如果输入是单个样本，去掉batch维度
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]

        info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}  # 构建信息字典
        return next_obs, rewards, terminals, info  # 返回下一状态、奖励、终止标志、附加信息


# ======================== MBPO (保持原始 episode 框架，内部使用官方组件) ========================

class MBPO:
    """
    MBPO（Model-Based Policy Optimization）算法实现
    结合模型学习和策略优化，通过短horizon的模型rollout生成虚拟数据增强训练
    """
    def __init__(self, env, agent, predict_env, env_pool, model_pool,
                 rollout_length, rollout_batch_size, real_ratio, num_episode,
                 model_train_freq=250, num_train_repeat=20,
                 policy_train_batch_size=256, max_path_length=1000,
                 log_dir='runs/MBPO', env_name=''):
        """
        初始化MBPO训练器
        参数:
            env: 真实环境
            agent: SAC智能体
            predict_env: 预测环境（封装的动力学模型）
            env_pool: 真实经验回放缓冲区
            model_pool: 模型生成的虚拟经验回放缓冲区
            rollout_length: 模型rollout的步长（horizon）
            rollout_batch_size: 每次rollout的初始状态数量
            real_ratio: 训练策略时使用真实数据的比例
            num_episode: 总训练回合数
            model_train_freq: 动力学模型训练频率（每多少步训练一次模型）
            num_train_repeat: 每次训练重复更新策略的次数
            policy_train_batch_size: 策略训练的批量大小
            max_path_length: 每个回合的最大步数
            log_dir: TensorBoard日志目录
        """
        self.env = env  # 真实环境
        self.agent = agent  # SAC智能体
        self.predict_env = predict_env  # 预测环境
        self.env_pool = env_pool  # 真实经验缓冲区
        self.model_pool = model_pool  # 模型生成的虚拟经验缓冲区
        self.rollout_length = rollout_length  # rollout步长
        self.rollout_batch_size = rollout_batch_size  # rollout批量大小
        self.real_ratio = real_ratio  # 真实数据比例
        self.num_episode = num_episode  # 总训练回合数
        self.model_train_freq = model_train_freq  # 动力学模型训练频率
        self.num_train_repeat = num_train_repeat  # 策略更新重复次数
        self.policy_train_batch_size = policy_train_batch_size  # 策略训练批量大小
        self.max_path_length = max_path_length  # 每回合最大步数
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M")  # 获取当前时间戳字符串
        env_suffix = f"_{env_name}" if env_name else ""
        self.writer = SummaryWriter(f"runs/MBPOv2_{ts}{env_suffix}")  # 创建TensorBoard日志记录器

    def rollout_model(self):
        """
        使用学习到的动力学模型进行rollout，生成虚拟经验数据
        返回:
            rollout_info: 包含rollout统计信息的字典
        """
        state, _, _, _, _ = self.env_pool.sample_all_batch(
            self.rollout_batch_size)

        total_samples = 0
        total_terminals = 0
        rollout_rewards = []
        rollout_model_disagreements = []

        for i in range(self.rollout_length):
            action = self.agent.select_action(state)
            next_states, rewards, terminals, info = self.predict_env.step(state, action)
            self.model_pool.push_batch([
                (state[j], action[j], rewards[j], next_states[j], terminals[j])
                for j in range(state.shape[0])
            ])

            total_samples += state.shape[0]
            total_terminals += terminals.sum()
            rollout_rewards.append(rewards)
            rollout_model_disagreements.append(info['dev'])

            nonterm_mask = ~terminals.squeeze(-1)
            if nonterm_mask.sum() == 0:
                break
            state = next_states[nonterm_mask]

        all_rewards = np.concatenate(rollout_rewards)
        all_disagreements = np.concatenate(rollout_model_disagreements)

        rollout_info = {
            'rollout_total_samples': total_samples,
            'rollout_terminal_ratio': float(total_terminals) / max(total_samples, 1),
            'rollout_reward_mean': float(np.mean(all_rewards)),
            'rollout_reward_std': float(np.std(all_rewards)),
            'rollout_model_disagreement_mean': float(np.mean(all_disagreements)),
            'rollout_model_disagreement_max': float(np.max(all_disagreements)),
            'rollout_effective_length': i + 1,
        }
        return rollout_info

    def update_agent(self):
        """
        更新SAC策略，混合使用真实数据和模型生成的虚拟数据
        返回:
            last_sac_info: 最后一次SAC更新的指标字典
        """
        env_batch_size = int(self.policy_train_batch_size * self.real_ratio)
        model_batch_size = self.policy_train_batch_size - env_batch_size
        last_sac_info = {}
        for epoch in range(self.num_train_repeat):
            env_state, env_action, env_reward, env_next_state, env_done = \
                self.env_pool.sample(env_batch_size)

            if model_batch_size > 0 and len(self.model_pool) > 0:
                model_state, model_action, model_reward, model_next_state, model_done = \
                    self.model_pool.sample_all_batch(model_batch_size)
                batch_state = np.concatenate((env_state, model_state), axis=0)
                batch_action = np.concatenate((env_action, model_action), axis=0)
                batch_reward = np.concatenate(
                    (np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0)
                batch_next_state = np.concatenate((env_next_state, model_next_state), axis=0)
                batch_done = np.concatenate(
                    (np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
            else:
                batch_state, batch_action, batch_reward = env_state, env_action, env_reward
                batch_next_state, batch_done = env_next_state, env_done

            batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
            batch_done = (~batch_done).astype(int)
            last_sac_info = self.agent.update_parameters(
                (batch_state, batch_action, batch_reward, batch_next_state, batch_done),
                self.policy_train_batch_size, epoch)

        return last_sac_info

    def train_model(self):
        """
        使用真实经验数据训练动力学模型
        返回:
            model_info: 包含模型训练指标的字典
        """
        state, action, reward, next_state, done = self.env_pool.sample(len(self.env_pool))
        delta_state = next_state - state
        inputs = np.concatenate((state, action), axis=-1)
        labels = np.concatenate(
            (np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)
        model_info = self.predict_env.model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)
        return model_info

    def explore(self):
        """
        使用当前策略进行一个完整回合的探索，收集真实经验
        返回:
            episode_return: 该回合的总奖励
        """
        reset_result = self.env.reset()  # 重置环境
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result  # 兼容新旧gym接口获取初始观测
        done, episode_return = False, 0  # 初始化终止标志和累计奖励
        while not done:  # 循环直到回合结束
            action = self.agent.select_action(obs)  # 用当前策略选择动作
            step_result = self.env.step(action)  # 在真实环境中执行动作
            if len(step_result) == 5:  # gymnasium新接口返回5个值
                next_obs, reward, terminated, truncated, _ = step_result  # 解包：观测、奖励、终止、截断、信息
                done = terminated or truncated  # 终止或截断都视为回合结束
            else:  # gym旧接口返回4个值
                next_obs, reward, done, _ = step_result  # 解包：观测、奖励、终止、信息
            self.env_pool.push(obs, action, reward, next_obs, done)  # 将经验存入真实缓冲区
            obs = next_obs  # 更新当前观测
            episode_return += reward  # 累计奖励
        return episode_return  # 返回回合总奖励

    def _log_sac_metrics(self, sac_info, global_step):
        """将SAC策略的训练指标写入TensorBoard"""
        self.writer.add_scalar('sac_loss/qf1_loss', sac_info['qf1_loss'], global_step)
        self.writer.add_scalar('sac_loss/qf2_loss', sac_info['qf2_loss'], global_step)
        self.writer.add_scalar('sac_loss/policy_loss', sac_info['policy_loss'], global_step)

        self.writer.add_scalar('sac_q/qf1_mean', sac_info['qf1_mean'], global_step)
        self.writer.add_scalar('sac_q/qf2_mean', sac_info['qf2_mean'], global_step)
        self.writer.add_scalar('sac_q/qf1_std', sac_info['qf1_std'], global_step)
        self.writer.add_scalar('sac_q/qf2_std', sac_info['qf2_std'], global_step)
        self.writer.add_scalar('sac_q/target_q_mean', sac_info['target_q_mean'], global_step)
        self.writer.add_scalar('sac_q/target_q_std', sac_info['target_q_std'], global_step)

        self.writer.add_scalar('sac_entropy/log_pi_mean', sac_info['log_pi_mean'], global_step)
        self.writer.add_scalar('sac_entropy/log_pi_std', sac_info['log_pi_std'], global_step)
        self.writer.add_scalar('sac_entropy/alpha', sac_info['alpha'], global_step)
        if 'alpha_loss' in sac_info:
            self.writer.add_scalar('sac_entropy/alpha_loss', sac_info['alpha_loss'], global_step)
            self.writer.add_scalar('sac_entropy/target_entropy', sac_info['target_entropy'], global_step)

        self.writer.add_scalar('sac_policy/mean_abs_action', sac_info['policy_mean_abs'], global_step)
        self.writer.add_scalar('sac_policy/action_std', sac_info['policy_std'], global_step)

        self.writer.add_scalar('sac_grad/policy_grad_norm', sac_info['policy_grad_norm'], global_step)
        self.writer.add_scalar('sac_grad/critic_grad_norm', sac_info['critic_grad_norm'], global_step)

        self.writer.add_scalar('sac_data/reward_mean', sac_info['reward_mean'], global_step)
        self.writer.add_scalar('sac_data/reward_std', sac_info['reward_std'], global_step)

    def _log_model_metrics(self, model_info, global_step):
        """将集成动力学模型的训练指标写入TensorBoard"""
        self.writer.add_scalar('model_train/epochs', model_info['model_train_epochs'], global_step)
        self.writer.add_scalar('model_train/train_loss_final', model_info['model_train_loss_final'], global_step)
        self.writer.add_scalar('model_train/data_size', model_info['model_train_data_size'], global_step)

        self.writer.add_scalar('model_holdout/mse_mean', model_info['model_holdout_mse_mean'], global_step)
        self.writer.add_scalar('model_holdout/mse_std', model_info['model_holdout_mse_std'], global_step)
        self.writer.add_scalar('model_holdout/mse_min', model_info['model_holdout_mse_min'], global_step)
        self.writer.add_scalar('model_holdout/mse_max', model_info['model_holdout_mse_max'], global_step)
        self.writer.add_scalar('model_holdout/elite_mse_mean', model_info['model_elite_mse_mean'], global_step)

        self.writer.add_scalar('model_var/max_logvar_mean', model_info['model_max_logvar_mean'], global_step)
        self.writer.add_scalar('model_var/min_logvar_mean', model_info['model_min_logvar_mean'], global_step)
        self.writer.add_scalar('model_var/holdout_var_mean', model_info['model_holdout_var_mean'], global_step)
        self.writer.add_scalar('model_var/holdout_var_max', model_info['model_holdout_var_max'], global_step)

        for i in range(self.predict_env.model.network_size):
            key = f'model_{i}_holdout_mse'
            if key in model_info:
                self.writer.add_scalar(f'model_individual/{key}', model_info[key], global_step)

    def _log_rollout_metrics(self, rollout_info, global_step):
        """将模型rollout的统计信息写入TensorBoard"""
        self.writer.add_scalar('rollout/total_samples', rollout_info['rollout_total_samples'], global_step)
        self.writer.add_scalar('rollout/terminal_ratio', rollout_info['rollout_terminal_ratio'], global_step)
        self.writer.add_scalar('rollout/reward_mean', rollout_info['rollout_reward_mean'], global_step)
        self.writer.add_scalar('rollout/reward_std', rollout_info['rollout_reward_std'], global_step)
        self.writer.add_scalar('rollout/model_disagreement_mean', rollout_info['rollout_model_disagreement_mean'], global_step)
        self.writer.add_scalar('rollout/model_disagreement_max', rollout_info['rollout_model_disagreement_max'], global_step)
        self.writer.add_scalar('rollout/effective_length', rollout_info['rollout_effective_length'], global_step)

    def train(self):
        """
        MBPO主训练循环
        先进行一次纯探索，然后交替进行：模型训练 -> 模型rollout -> 真实交互 -> 策略更新
        返回:
            return_list: 每个回合的累计奖励列表
        """
        return_list = []
        global_step = 0
        model_train_count = 0

        explore_return = self.explore()
        print('episode: 1, return: %d' % explore_return)
        return_list.append(explore_return)
        self.writer.add_scalar('reward/episode_return', explore_return, 1)
        self.writer.add_scalar('buffer/env_pool_size', len(self.env_pool), 1)

        for i_episode in range(self.num_episode - 1):
            reset_result = self.env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            done, episode_return = False, 0
            step = 0
            episode_rewards = []

            while not done:
                if step % self.model_train_freq == 0:
                    model_info = self.train_model()
                    rollout_info = self.rollout_model()
                    model_train_count += 1
                    self._log_model_metrics(model_info, model_train_count)
                    self._log_rollout_metrics(rollout_info, model_train_count)

                action = self.agent.select_action(obs)
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    next_obs, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    next_obs, reward, done, _ = step_result
                self.env_pool.push(obs, action, reward, next_obs, done)
                obs = next_obs
                episode_return += reward
                episode_rewards.append(reward)

                sac_info = self.update_agent()
                global_step += 1

                if global_step % 100 == 0:
                    self._log_sac_metrics(sac_info, global_step)
                    self.writer.add_scalar('buffer/env_pool_size', len(self.env_pool), global_step)
                    self.writer.add_scalar('buffer/model_pool_size', len(self.model_pool), global_step)

                step += 1

            return_list.append(episode_return)
            episode_num = i_episode + 2

            self.writer.add_scalar('reward/episode_return', episode_return, episode_num)
            self.writer.add_scalar('reward/episode_length', step, episode_num)
            self.writer.add_scalar('reward/episode_reward_mean', float(np.mean(episode_rewards)), episode_num)
            self.writer.add_scalar('reward/episode_reward_std', float(np.std(episode_rewards)), episode_num)
            self.writer.add_scalar('reward/episode_reward_min', float(np.min(episode_rewards)), episode_num)
            self.writer.add_scalar('reward/episode_reward_max', float(np.max(episode_rewards)), episode_num)

            if len(return_list) >= 10:
                self.writer.add_scalar('reward/return_ma10', float(np.mean(return_list[-10:])), episode_num)
            if len(return_list) >= 50:
                self.writer.add_scalar('reward/return_ma50', float(np.mean(return_list[-50:])), episode_num)

            print('episode: %d, return: %d, steps: %d, global_step: %d' %
                  (episode_num, episode_return, step, global_step))

        self.writer.close()
        return return_list


def set_seed(seed=42):
    """
    设置全局随机种子，确保实验可复现
    参数:
        seed: 随机种子值
    """
    random.seed(seed)  # 设置Python标准库random的种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch CPU的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置PyTorch所有GPU的随机种子
    torch.backends.cudnn.deterministic = True  # 设置cuDNN使用确定性算法（保证可复现）
    torch.backends.cudnn.benchmark = False  # 关闭cuDNN的benchmark模式（benchmark会自动选择最快算法，但不确定性）


# ======================== 主函数 ========================
if __name__ == "__main__":
    seed = 42  # 设置随机种子
    set_seed(seed)  # 应用随机种子

    env_name = 'Walker2d-v4'  # 选择环境：倒立摆（Pendulum）Pendulum-v1\HalfCheetah-v5\Hopper-v4\Walker2d-v4
    env = gym.make(env_name)  # 创建gym环境实例
    env.reset(seed=seed)  # 重置环境并设置种子
    env.action_space.seed(seed)  # 设置动作空间的随机种子

    num_episodes = 100  # 总训练回合数
    real_ratio = 0.05  # 真实数据占比5%（95%使用模型生成的虚拟数据）
    buffer_size = 10000  # 真实经验缓冲区大小
    rollout_batch_size = 1000  # 每次rollout采样1000个初始状态
    rollout_length = 1  # 每次rollout只走1步（短horizon rollout）
    model_pool_size = rollout_batch_size * rollout_length  # 模型缓冲区大小 = rollout批量 × rollout步长

    state_dim = env.observation_space.shape[0]  # 获取状态空间维度
    action_dim = env.action_space.shape[0]  # 获取动作空间维度

    class Args:
        """SAC算法的超参数配置类"""
        gamma = 0.98  # 折扣因子
        tau = 0.005  # 目标网络软更新系数
        alpha = 0.2  # 初始熵系数（温度参数）
        target_update_interval = 1  # 目标网络更新间隔（每步都更新）
        automatic_entropy_tuning = True  # 启用自动熵调整
        hidden_size = 128  # 策略和Q网络的隐藏层大小
        lr = 5e-4  # 学习率
    args = Args()  # 创建超参数实例

    num_networks = 7  # 集成模型总数
    num_elites = 5  # 精英模型数量

    agent = SAC(state_dim, env.action_space, args)  # 创建SAC智能体
    env_model = EnsembleDynamicsModel(num_networks, num_elites,  # 创建集成动力学模型
                                      state_dim, action_dim,
                                      reward_size=1, hidden_size=200,
                                      use_decay=True)
    predict_env = PredictEnv(env_model, env_name)  # 创建预测环境（封装动力学模型）
    env_pool = ReplayMemory(buffer_size)  # 创建真实经验回放缓冲区
    model_pool = ReplayMemory(model_pool_size)  # 创建模型生成的虚拟经验回放缓冲区

    mbpo = MBPO(env, agent, predict_env, env_pool, model_pool,  # 创建MBPO训练器
                rollout_length, rollout_batch_size, real_ratio, num_episodes,
                model_train_freq=50, num_train_repeat=10,  # 每50步训练一次动力学模型，每步SAC策略进行10次梯度更新
                policy_train_batch_size=64, max_path_length=1000,
                env_name=env_name)  # 策略训练批量64，每回合最多1000步

    return_list = mbpo.train()  # 开始MBPO训练，返回每回合的奖励列表

    # ---- 绘制训练曲线 ----
    episodes_list = list(range(len(return_list)))  # 生成回合编号列表
    plt.plot(episodes_list, return_list)  # 绘制奖励曲线
    plt.xlabel('Episodes')  # x轴标签：回合数
    plt.ylabel('Returns')  # y轴标签：累计奖励
    plt.title('MBPO on {}'.format(env_name))  # 图表标题
    plt.show()  # 显示图表
