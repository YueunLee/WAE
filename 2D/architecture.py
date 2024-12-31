import torch
from torch import nn, autograd
from torch.nn import functional as F

class MLPBlock(nn.Module):
    def __init__(self, 
                input_dim: int,
                output_dim: int,
                hidden_dim: int, 
                num_layers: int,
                last_activation: bool,                
                batchnorm: bool=False,
                activation: nn.Module = nn.Sigmoid,
                seed: int=0):
        
        super().__init__()
        encoder_list = []
        
        assert num_layers >= 1
        if num_layers == 1:
            encoder_list.append(
                nn.Linear(input_dim, output_dim)
            )
        else: # num_layers >= 2
            encoder_list.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if batchnorm else nn.Identity(),
                activation(),
            ])
            for _ in range(num_layers-2):
                encoder_list.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim) if batchnorm else nn.Identity(),
                    activation(),
                ])
            encoder_list.extend([
                nn.Linear(hidden_dim, output_dim),
            ])
        
        if last_activation:
            encoder_list.append(activation())
            
        self.model = nn.Sequential(*encoder_list)
        self.seed = seed
        self.reset()

    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def reset(self, train_seed:int=None):
        if train_seed is None:
            train_seed = self.seed
        torch.manual_seed(train_seed)
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

"""
    Input Convex Neural Network (ICNN); its gradient is used for an optimal transport map

    Original source code:
        https://github.com/iamalexkorotin/Wasserstein2GenerativeNetworks
        https://github.com/iamalexkorotin/Wasserstein2Benchmark
"""
class ConvexLinear(nn.Module):
    '''Convex Linear Layer'''
    __constants__ = ['in_features', 'out_features', 'weight', 'bias']

    def __init__(self, in_features, out_features, bias=True, rank=1):
        super(ConvexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.weight = nn.Parameter(torch.Tensor(
            torch.randn(out_features, in_features)
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        device = self.weight.get_device()
        return F.linear(input.to(device), self.weight, self.bias).type_as(input)
    
class DenseICNN(nn.Module):
    '''Fully Conncted ICNN with input-quadratic skip connections.'''
    def __init__(
        self, dim,
        hidden_sizes=[32, 32, 32],
        rank=1, activation='celu',
        strong_convexity=1e-6,
        batch_size=1024,
        weights_init_std=0.1,
    ):
        super(DenseICNN, self).__init__()

        self.dim = dim
        self.strong_convexity = strong_convexity
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.rank = rank
        self.batch_size = batch_size

        #
        self.linear_layers = nn.ModuleList([
            ConvexLinear(dim, out_features, rank=rank, bias=True)
            for out_features in hidden_sizes
        ])

        sizes = zip(hidden_sizes[:-1], hidden_sizes[1:])
        self.convex_layers = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=False)
            for (in_features, out_features) in sizes
        ])

        self.final_layer = nn.Linear(hidden_sizes[-1], 1, bias=False)

        self.weights_init_std = weights_init_std
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            p.data = (torch.randn(p.shape, dtype=torch.float32) * self.weights_init_std).to(p)

    def forward(self, input):
        '''Evaluation of the discriminator value. Preserves the computational graph.'''
        device = self.linear_layers[0].weight.get_device()
        output = self.linear_layers[0](input.to(device))
        for linear_layer, convex_layer in zip(self.linear_layers[1:], self.convex_layers):
            output = convex_layer(output) + linear_layer(input.to(device))
            if self.activation == 'celu':
                output = torch.celu(output)
            elif self.activation == 'softplus':
                output = F.softplus(output)
            elif self.activation == 'relu':
                output = F.relu(output)
            else:
                raise Exception('Activation is not specified or unknown.')

        return self.final_layer(output).type_as(input) + .5 * self.strong_convexity * (input ** 2).sum(dim=1).reshape(-1, 1)

    def push(self, input):
        device = self.linear_layers[0].weight.get_device()
        if len(input) <= self.batch_size:
            input_dev = input.to(device)
            output = autograd.grad(
                outputs=self.forward(input_dev), inputs=input_dev,
                create_graph=True, retain_graph=True,
                only_inputs=True,
                grad_outputs=torch.ones_like(input_dev[:, :1], requires_grad=False)
            )[0].type_as(input)
        else:
            output = torch.zeros_like(input, requires_grad=False)
            for i in range(0, len(input), self.batch_size):
                input_batch = input[i:i+self.batch_size]
                input_batch_dev = input_batch.to(device)
                
                output.data[i:i+self.batch_size] = autograd.grad(
                    outputs=self.forward(input_batch_dev), inputs=input_batch_dev,
                    create_graph=False, retain_graph=False,
                    only_inputs=True,
                    grad_outputs=torch.ones_like(input_batch_dev[:, :1], requires_grad=False)
                )[0].type_as(input_batch).data
        return output

    # def push(self, input, create_graph=True, retain_graph=True):
    #     '''
    #     Pushes input by using the gradient of the network. By default preserves the computational graph.
    #     Apply to small batches.
    #     '''
    #     assert len(input) <= self.batch_size
    #     device = self.quadratic_layers[0].weight.get_device()
    #     input_dev = input.to(device)
    #     output = autograd.grad(
    #         outputs=self.forward(input_dev), inputs=input_dev,
    #         create_graph=create_graph, retain_graph=retain_graph,
    #         only_inputs=True,
    #         grad_outputs=torch.ones_like(input_dev[:, :1], requires_grad=False)
    #     )[0]
    #     return output.type_as(input)

    # def push_nograd(self, input):
    #     '''
    #     Pushes input by using the gradient of the network. Does not preserve the computational graph.
    #     Use for pushing large batches (the function uses minibatches).
    #     '''
    #     output = torch.zeros_like(input, requires_grad=False)
    #     for i in range(0, len(input), self.batch_size):
    #         input_batch = input[i:i+self.batch_size]
    #         output.data[i:i+self.batch_size] = self.push(
    #             input_batch, create_graph=False, retain_graph=False
    #         ).data
    #     return output

    def convexify(self):
        for layer in self.convex_layers:
            if (isinstance(layer, nn.Linear)):
                layer.weight.data.clamp_(0)
        self.final_layer.weight.data.clamp_(0)


"""
    RealNVP; Bijective Neural Network

    Original codes: 
        https://github.com/chrischute/real-nvp
        https://github.com/siihwanpark/Real-NVP
"""
class CouplingLayer(nn.Module):
    # dim: input & output dimension
    # identity: 0 (x1) or 1 (x2);
    # Coupling: split half of input dimension
    def __init__(self, dim, hidden_size = 32, identity=0):
        super(CouplingLayer, self).__init__()

        d = dim // 2
        self.dim = dim
        self.split_idx = d
        self.identity = identity
        
        in_features = d if self.identity == 0 else (dim - d)
        out_features = (dim - d) if self.identity == 0 else d

        # if len(hidden_size) == 0:
        #     st_layers_lst = [
        #         nn.Linear(in_features, 2 * out_features)
        #     ]
        # else:
        #     st_layers_lst = [
        #         nn.Linear(in_features, hidden_sizes[0]),
        #         nn.LeakyReLU(negative_slope=0.02, inplace=True),
        #     ]

        #     for i in range(len(hidden_sizes)-1):
        #         st_layers_lst.extend([
        #             nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
        #             nn.LeakyReLU(negative_slope=0.02, inplace=True),
        #         ])
            
        #     st_layers_lst.append(
        #         nn.Linear(hidden_sizes[-1], 2 * out_features)
        #     )
        st_layers_lst = [
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(inplace=True),

            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(inplace=True),

            nn.Linear(hidden_size, 2 * out_features)
        ]
        
        self.st_net = nn.Sequential(*st_layers_lst)

    def forward(self, x, reverse=False):
        device = self.st_net[0].weight.get_device()
        x_dev = x.to(device)
        x1, x2 = x_dev[:, :self.split_idx], x_dev[:, self.split_idx:]
        
        a, b = (x1, x2) if self.identity == 0 else (x2, x1)

        st = self.st_net(a)
        s, t = st.split(st.size(1) // 2, dim=1)
        if reverse:
            inv_exp_s = s.mul(-1.0).exp()
            b = (b - t) * inv_exp_s
        else:
            exp_s = s.exp()
            b = b * exp_s + t

        if self.identity != 0:
            return torch.cat([b, a], dim=1).type_as(x)
        return torch.cat([a, b], dim=1).type_as(x)
    
class RealNVP(nn.Module):
    def __init__(self, dim, num_layers, hidden_size, weights_init_std=0.1):
        super(RealNVP, self).__init__()
        
        self.model = nn.ModuleList(
            CouplingLayer(dim=dim, hidden_size=hidden_size, identity=(i % 2)) for i in range(num_layers)
        )

        self.weights_init_std = weights_init_std
        self._init_weights()

    def _init_weights(self):
        for p in self.model.parameters():
            p.data = (torch.randn(p.shape, dtype=torch.float32) * self.weights_init_std).to(p)

    def forward(self, x, reverse=False):
        model_idx_lst = range(len(self.model))
        if reverse:
            model_idx_lst = reversed(model_idx_lst)

        for i in model_idx_lst:
            x = self.model[i](x, reverse)

        return x