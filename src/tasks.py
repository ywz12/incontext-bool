import math

import torch
import numpy as np
import pdb
from scipy.stats import ortho_group

def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    '''
    ys_pred: {-inf, inf}
    ys: {-1, 1}
    '''
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=0, **kwargs
):
    task_names_to_classes = {
        "teach_tdhomogenous":teach_TDhomogenous,
        "teach_pbtdhomogenous0":teach_PBTDhomogenous0,
        "pbtdhomogenous0":PBTDhomogenous0,
        "teach_pbtdinhomogenousb":teach_PBTDinhomogenousb,
        "pbtdinhomogenousb":PBTDinhomogenousb,
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
        # "conjunction": Conjunction,
        # "mono_conjunction": MonoConjunction,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks > 0:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError



class teach_TDhomogenous(Task):
    def __init__(self, n_dims, b_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(teach_TDhomogenous, self).__init__(n_dims, b_size, pool_dict, seeds)
        self.scale = scale

        # Define w_b based on pool_dict and seeds, similar to LinearRegression
        if pool_dict is None and seeds is None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1, dtype=torch.float)

            for i in range(self.b_size):
                # 随机选择一个维度，将该维度置为1，其余维度为0
                self.w_b[i, 0, 0] = 1.0

        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                # 根据种子随机选择一个维度，将该维度置为1
                random_index = torch.randint(0, self.n_dims, (1,), generator=generator)
                self.w_b[i, random_index, 0] = 1.0
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:b_size]
            self.w_b = pool_dict["w"][indices]

    def sample_xs(self, n_points, b_size):
        """Sample xs for each batch."""
        xs_b = []
        for i in range(b_size):
            #A = self.A_b[i]
            e1 = torch.zeros(self.n_dims)
            e1[0] = 1

            e_is = torch.eye(self.n_dims)[1:]
            u = -e_is.sum(dim=0)

            #A_T_e1 = A.T @ e1
            #A_T_e1 = A_T_e1.unsqueeze(0).unsqueeze(0)

            #A_T_eis = (A.T @ e_is.T).unsqueeze(0).unsqueeze(0)
            #A_T_u = (A.T @ u).unsqueeze(0).unsqueeze(0)

            T = torch.cat([e_is, u, e1], dim=1)
            T_size = T.shape[1]
            num_random_points = n_points - T_size
            random_xs = torch.randn(num_random_points, self.n_dims)  # 剩下部分随机生成

        # 将 T 和随机生成的 xs 结合起来
            xs = torch.cat([T.squeeze(0), random_xs], dim=0)  # 组合 T 和随机部分
            xs_b.append(xs.unsqueeze(0))

        xs_b = torch.cat(xs_b, dim=0)
        return xs_b

    def evaluate(self, xs_b):
        """Evaluate xs_b using the batch of w_b."""
        ys_b = torch.zeros(xs_b.size(0), xs_b.size(1))
        for i in range(xs_b.size(0)):
            w = self.w_b[i].to(xs_b.device)
            ys_b[i] = (xs_b[i] @ w).squeeze().sign()
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        w = torch.zeros(num_tasks, n_dims, 1)
        for i in range(num_tasks):
            # 随机选择一个维度，将该维度置为1
            #random_index = torch.randint(0, n_dims, (1,))
            w[i, 0, 0] = 1.0
        return {"w": w}

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy


class teach_PBTDhomogenous0(Task):
    def __init__(self, n_dims, b_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(teach_PBTDhomogenous0, self).__init__(n_dims, b_size, pool_dict, seeds)
        self.scale = scale

        # Define w_b based on pool_dict and seeds, but using the modified logic for the last two dimensions
        if pool_dict is None and seeds is None:
            # 直接生成 (self.b_size, self.n_dims, 1) 大小的权重矩阵
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1, dtype=torch.float)
            
            for i in range(self.b_size):
                # 随机生成最后一维在 [0, 1] 之间
                last_dim = torch.rand(1)
                # 倒数第二维为 sqrt(1 - 最后一维的平方)
                second_last_dim = torch.sqrt(1 - last_dim ** 2)
                # 将倒数两维填入 w_b，其他维度为 0
                self.w_b[i, -1, 0] = last_dim
                self.w_b[i, -2, 0] = second_last_dim
                # 其余维度保持为 0
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                # 随机生成最后一维在 [0, 1] 之间
                last_dim = torch.rand(1, generator=generator)
                # 倒数第二维为 sqrt(1 - 最后一维的平方)
                second_last_dim = torch.sqrt(1 - last_dim ** 2)
                # 将倒数两维填入 w_b，其他维度为 0
                self.w_b[i, -1, 0] = last_dim
                self.w_b[i, -2, 0] = second_last_dim
                # 其余维度保持为 0
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:b_size]
            self.w_b = pool_dict["w"][indices]

    def sample_xs(self, n_points, b_size):
        xs_b = []
        for i in range(b_size):
          
            e_d = torch.zeros(self.n_dims)
            e_d[-1] = 1

            # First sample based on w_d
            if self.w_b[i, -1, 0] > 0:
                first_sample =  (-e_d)
            else:
                first_sample =  e_d

            # Second sample
            w_star = self.w_b[i].squeeze()
            second_sample =  torch.cat([torch.zeros(self.n_dims - 2), w_star[-1].unsqueeze(0), -w_star[-2].unsqueeze(0)])

            T = torch.stack([first_sample, second_sample], dim=0)

            # Calculate the number of random samples needed
            T_size = T.size(0)
            num_random_points = n_points - T_size
        
            # Generate random samples for the remaining points
            random_xs = torch.randn(num_random_points, self.n_dims) 

            # Combine T and random samples
            xs = torch.cat([T, random_xs], dim=0)

            xs_b.append(xs.unsqueeze(0))

        # Stack all batches
        xs_b = torch.cat(xs_b, dim=0)
        return xs_b

    def evaluate(self, xs_b):
        """Evaluate xs_b using the batch of w_b."""
        ys_b = torch.zeros(xs_b.size(0), xs_b.size(1))
        for i in range(xs_b.size(0)):
            w = self.w_b[i].to(xs_b.device)
            ys_b[i] = (xs_b[i] @ w).squeeze().sign()
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        """Generate a pool dictionary with modified w values."""
        w = torch.zeros(num_tasks, n_dims, 1)
        for i in range(num_tasks):
            # 随机生成最后一维在 [0, 1] 之间
            last_dim = torch.rand(1)
            # 倒数第二维为 sqrt(1 - 最后一维的平方)
            second_last_dim = torch.sqrt(1 - last_dim ** 2)
            # 将倒数两维填入 w_b，其他维度为 0
            w[i, -1, 0] = last_dim
            w[i, -2, 0] = second_last_dim
            # 其余维度保持为 0
        return {"w": w}

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy

class PBTDhomogenous0(Task):
    def __init__(self, n_dims, b_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(PBTDhomogenous0, self).__init__(n_dims, b_size, pool_dict, seeds)
        self.scale = scale

        # Define w_b based on pool_dict and seeds, but using the modified logic for the last two dimensions
        if pool_dict is None and seeds is None:
            # 直接生成 (self.b_size, self.n_dims, 1) 大小的权重矩阵
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1, dtype=torch.float)
            
            for i in range(self.b_size):
                # 随机生成最后一维在 [0, 1] 之间
                last_dim = torch.rand(1)
                # 倒数第二维为 sqrt(1 - 最后一维的平方)
                second_last_dim = torch.sqrt(1 - last_dim ** 2)
                # 将倒数两维填入 w_b，其他维度为 0
                self.w_b[i, -1, 0] = last_dim
                self.w_b[i, -2, 0] = second_last_dim
                # 其余维度保持为 0
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                # 随机生成最后一维在 [0, 1] 之间
                last_dim = torch.rand(1, generator=generator)
                # 倒数第二维为 sqrt(1 - 最后一维的平方)
                second_last_dim = torch.sqrt(1 - last_dim ** 2)
                # 将倒数两维填入 w_b，其他维度为 0
                self.w_b[i, -1, 0] = last_dim
                self.w_b[i, -2, 0] = second_last_dim
                # 其余维度保持为 0
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:b_size]
            self.w_b = pool_dict["w"][indices]

    def sample_xs(self, n_points, b_size):
        """Sample xs for each batch, similar to TDhomogeneous."""
        xs_b = []
        for i in range(b_size):
            xs = torch.randn(n_points, self.n_dims) 

            xs_b.append(xs.unsqueeze(0))

        # Stack all batches
        xs_b = torch.cat(xs_b, dim=0)
        return xs_b

    def evaluate(self, xs_b):
        """Evaluate xs_b using the batch of w_b."""
        ys_b = torch.zeros(xs_b.size(0), xs_b.size(1))
        for i in range(xs_b.size(0)):
            w = self.w_b[i].to(xs_b.device)
            ys_b[i] = (xs_b[i] @ w).squeeze().sign()
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        """Generate a pool dictionary with modified w values."""
        w = torch.zeros(num_tasks, n_dims, 1)
        for i in range(num_tasks):
            # 随机生成最后一维在 [0, 1] 之间
            last_dim = torch.rand(1)
            # 倒数第二维为 sqrt(1 - 最后一维的平方)
            second_last_dim = torch.sqrt(1 - last_dim ** 2)
            # 将倒数两维填入 w_b，其他维度为 0
            w[i, -1, 0] = last_dim
            w[i, -2, 0] = second_last_dim
            # 其余维度保持为 0
        return {"w": w}

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy

class teach_PBTDinhomogenousb(Task):
    def __init__(self, n_dims, b_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(teach_PBTDinhomogenousb, self).__init__(n_dims, b_size, pool_dict, seeds)
        self.scale = scale

        # 定义 w_b 基于 pool_dict 和 seeds
        if pool_dict is None and seeds is None:
            # 直接生成 (self.b_size, self.n_dims, 1) 大小的权重矩阵，满足最后三维规则
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1, dtype=torch.float)
            
            for i in range(self.b_size):
                # 最后一维在 (0, 1) 内
                last_dim = torch.rand(1).item()

                # 倒数第二维在 (0, 1 - 最后一维) 内
                second_last_dim = torch.rand(1).item() * (1 - last_dim)

                # 倒数第三维为 sqrt(1 - 最后两维的平方和)
                third_last_dim = torch.sqrt(torch.tensor(1 - last_dim**2 - second_last_dim**2)).item()

                # 将这些值赋给最后三维
                self.w_b[i, -1, 0] = last_dim
                self.w_b[i, -2, 0] = second_last_dim
                self.w_b[i, -3, 0] = third_last_dim

        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)

                # 随机生成遵循规则的最后三维
                last_dim = torch.rand(1, generator=generator).item()
                second_last_dim = torch.rand(1, generator=generator).item() * (1 - last_dim)
                third_last_dim = torch.sqrt(torch.tensor(1 - last_dim**2 - second_last_dim**2)).item()

                # 将这些值赋给最后三维
                self.w_b[i, -1, 0] = last_dim
                self.w_b[i, -2, 0] = second_last_dim
                self.w_b[i, -3, 0] = third_last_dim

        else:
            # 使用 pool_dict 中的 w 值
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:b_size]
            self.w_b = pool_dict["w"][indices]

        # 为每个任务生成偏差 b_b，取值范围为 {1, -1}
        self.b_b = torch.randint(0, 2, (self.b_size,)) * 2 - 1

     

    def sample_xs(self, n_points, b_size):
        """Sample xs for each batch."""
        xs_b = []
        for i in range(b_size):
            #A = self.A_b[i]
            e_d = torch.zeros(self.n_dims)
            e_d[-1] = 1

            # Sample a_1  (0, 0, ..., 0)
            a_1 = torch.zeros(1, self.n_dims)

            # Sample a_2 (-\frac{2b^*}{w_d^*} \cdot e_d)
            a_2 = (-2 * self.b_b[i] / self.w_b[i][-1].item()) * e_d

            # Sample a_3 (-\frac{b^*}{w_d^*} \cdot e_d)
            a_3 = (-self.b_b[i] / self.w_b[i][-1].item()) * e_d

            #sample 4
            e_d_1 = torch.zeros(self.n_dims-1)
            e_d_1[-1] = 1
            if self.w_b[i, -2, 0] > 0:
                a_4 = torch.cat([ -e_d_1, -self.b_b[i] / self.w_b[i][-1].item()]).unsqueeze(0)
            else:
                a_4 = torch.cat([ e_d_1, -self.b_b[i] / self.w_b[i][-1].item()]).unsqueeze(0)
            # Sample a_5
            a_5 = torch.cat([torch.zeros(self.n_dims - 3), self.w_b[i][-2].unsqueeze(0), -self.w_b[i][-3].unsqueeze(0), 
                            -self.b_b[i] / self.w_b[i][-1].item()]).unsqueeze(0)

            # Sample a_6
            a_6 = torch.cat([torch.sign(self.w_b[i][:-1]), 
                             -((torch.norm(self.w_b[i][-2]) + self.b_b[i]) / self.w_b[i][-1].item())]).unsqueeze(0)

            T = torch.stack([a_1, a_2.unsqueeze(0), a_3.unsqueeze(0), a_4, a_5, a_6], dim=0)

            # Calculate the number of random samples needed
            T_size = T.size(0)
            num_random_points = n_points - T_size
        
            # Generate random samples for the remaining points
            random_xs = torch.randn(num_random_points, self.n_dims)  # Randomly generate remaining samples

            # Combine T and random samples
            xs = torch.cat([T.squeeze(0), random_xs], dim=0)

            xs_b.append(xs.unsqueeze(0))

        # Stack all batches
        xs_b = torch.cat(xs_b, dim=0)
        return xs_b

    def evaluate(self, xs_b):
        """Evaluate xs_b using the batch of w_b."""
        ys_b = torch.zeros(xs_b.size(0), xs_b.size(1))
        for i in range(xs_b.size(0)):
            w = self.w_b[i].to(xs_b.device)
            b = self.b_b[i].to(xs_b.device)
            ys_b[i] = (xs_b[i] @ w).squeeze() + b
        return ys_b.sign()

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        """Generate a pool dictionary with w values where only the last three dimensions are non-zero."""
        w = torch.zeros(num_tasks, n_dims, 1)  # Initialize w as all zeros

        for i in range(num_tasks):
          # 1. 随机生成最后一维在 (0, 1) 之间
          last_dim = torch.rand(1).item()
          # 2. 倒数第二维在 (0, 1 - 最后一维) 之间生成
          second_last_dim = torch.rand(1).item() * (1 - last_dim)
          # 3. 倒数第三维为 sqrt(1 - 最后两维的平方和)
          third_last_dim = torch.sqrt(torch.tensor(1 - last_dim**2 - second_last_dim**2)).item()
          # 将这三个维度的值赋给最后三维
          w[i, -1, 0] = last_dim
          w[i, -2, 0] = second_last_dim
          w[i, -3, 0] = third_last_dim

        # 其余维度默认已经是 0，无需修改

        return {"w": w}

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy

class PBTDinhomogenousb(Task):
    def __init__(self, n_dims, b_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(PBTDinhomogenousb, self).__init__(n_dims, b_size, pool_dict, seeds)
        self.scale = scale

        # 定义 w_b 基于 pool_dict 和 seeds
        if pool_dict is None and seeds is None:
            # 直接生成 (self.b_size, self.n_dims, 1) 大小的权重矩阵，满足最后三维规则
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1, dtype=torch.float)
            
            for i in range(self.b_size):
                # 最后一维在 (0, 1) 内
                last_dim = torch.rand(1).item()

                # 倒数第二维在 (0, 1 - 最后一维) 内
                second_last_dim = torch.rand(1).item() * (1 - last_dim)

                # 倒数第三维为 sqrt(1 - 最后两维的平方和)
                third_last_dim = torch.sqrt(torch.tensor(1 - last_dim**2 - second_last_dim**2)).item()

                # 将这些值赋给最后三维
                self.w_b[i, -1, 0] = last_dim
                self.w_b[i, -2, 0] = second_last_dim
                self.w_b[i, -3, 0] = third_last_dim

        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)

                # 随机生成遵循规则的最后三维
                last_dim = torch.rand(1, generator=generator).item()
                second_last_dim = torch.rand(1, generator=generator).item() * (1 - last_dim)
                third_last_dim = torch.sqrt(torch.tensor(1 - last_dim**2 - second_last_dim**2)).item()

                # 将这些值赋给最后三维
                self.w_b[i, -1, 0] = last_dim
                self.w_b[i, -2, 0] = second_last_dim
                self.w_b[i, -3, 0] = third_last_dim

        else:
            # 使用 pool_dict 中的 w 值
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:b_size]
            self.w_b = pool_dict["w"][indices]

        # 为每个任务生成偏差 b_b，取值范围为 {1, -1}
        self.b_b = torch.randint(0, 2, (self.b_size,)) * 2 - 1

     

    def sample_xs(self, n_points, b_size):
        """Sample xs for each batch."""
        xs_b = []
        for i in range(b_size):
            # Generate random samples for the remaining points
            xs = torch.randn(n_points, self.n_dims)  # Randomly generate remaining samples
            xs_b.append(xs.unsqueeze(0))

        # Stack all batches
        xs_b = torch.cat(xs_b, dim=0)
        return xs_b

    def evaluate(self, xs_b):
        """Evaluate xs_b using the batch of w_b."""
        ys_b = torch.zeros(xs_b.size(0), xs_b.size(1))
        for i in range(xs_b.size(0)):
            w = self.w_b[i].to(xs_b.device)
            b = self.b_b[i].to(xs_b.device)
            ys_b[i] = (xs_b[i] @ w).squeeze() + b
        return ys_b.sign()

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        """Generate a pool dictionary with w values where only the last three dimensions are non-zero."""
        w = torch.zeros(num_tasks, n_dims, 1)  # Initialize w as all zeros

        for i in range(num_tasks):
          # 1. 随机生成最后一维在 (0, 1) 之间
          last_dim = torch.rand(1).item()
          # 2. 倒数第二维在 (0, 1 - 最后一维) 之间生成
          second_last_dim = torch.rand(1).item() * (1 - last_dim)
          # 3. 倒数第三维为 sqrt(1 - 最后两维的平方和)
          third_last_dim = torch.sqrt(torch.tensor(1 - last_dim**2 - second_last_dim**2)).item()
          # 将这三个维度的值赋给最后三维
          w[i, -1, 0] = last_dim
          w[i, -2, 0] = second_last_dim
          w[i, -3, 0] = third_last_dim

        # 其余维度默认已经是 0，无需修改

        return {"w": w}

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy



class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False
            w[mask] = 0

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearClassification(LinearRegression):
    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy


class NoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy


class QuadraticRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b**2) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad


class Relu2nnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=4,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.W2[i] = torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class DecisionTree(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, depth=4):

        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.depth = depth

        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = torch.randn(self.dt_tensor.shape)
        elif seeds is not None:
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(dt_tensor)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=2 ** (depth + 1) - 1,
                    generator=generator,
                )
                self.target_tensor[i] = torch.randn(
                    self.dt_tensor[i].shape, generator=generator
                )
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

