"""### Layer"""

# GWNET
class nconv2(nn.Module):
    def __init__(self):
        super(nconv2,self).__init__()

    def forward(self,x, A):
        # x: torch.Size([64, 32, 207, 12])
        # A: torch.Size([207, 207])
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)



# Process the input with the Adj matrix #
class gcn(nn.Module):
    def __init__(self,model_type, adj_mx_len,  c_in,c_out,gdep,dropout,alpha):
        super(gcn, self).__init__()

        self.model_type = model_type
        self.adj_mx_len = adj_mx_len

        # 這裡要實驗看看
        if self.model_type == "GWNET":
            self.nconv = nconv2()

        c_in = (gdep*adj_mx_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):


        h = x
        out = [h]

        if self.model_type == "GWNET":
            # support: 儲存adj matrix/transpose adj matrix/adaptive matrix(optional)
            for a in adj:
                # 執行diffision convl 第1層
                x1 = self.nconv(x,a)                                  # 將各sensor的值透過adj matrix進行聚合
                out.append(x1)

                # 執行diffusion convl 第k層 (1<k<=2)
                # Paper的diffusion step= 2
                # k=0: P*a=x1
                # k=1: P*(x1)=P*(P*a)=P^2*a
                for k in range(2, self.gdep + 1):
                    x2 = self.nconv(x1,a)
                    out.append(x2)
                    x1 = x2

        # 問題: 原論文是將每個H經過MLP後, 最後進行加總
        # 可能因為過多MLP層, 故作者將H先串接, 最後將dim=1展開, 再經過MLP
        # [1+depth layer] -- cat -> ([64, 96, 207, 13]) -- MLP-> ([64, 32, 207, 13])
        ho = torch.cat(out,dim=1)   # torch.Size([64, 96, 207, ?])
        ho = self.mlp(ho)

        # Paper step: information propagation step -- END #
        return ho


# dilated_inception: 有[2,3,6,7]4個平行層 => 分別處理後再串接 #
class dilated_inception(nn.Module):
    def __init__(self, kernel_set, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = kernel_set
        cout = int(cout/len(self.kernel_set))    # 32/4 = 8

        # (1x2) & (1x3) & (1x6) & (1x7)
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

    def forward(self,input):
        if args.log_print:
            print("# dilated_inception input", input.shape)

        x = []

        # 每層拿"原"input, 而非處理後的input
        # ex:
        # kernel_set: 2 -> torch.Size([64, 8, 207, 18])
        # kernel_set: 3 -> torch.Size([64, 8, 207, 17])
        # kernel_set: 6 -> torch.Size([64, 8, 207, 14])
        # kernel_set: 7 -> torch.Size([64, 8, 207, 13])
        for i in range(len(self.kernel_set)):

            x.append(self.tconv[i](input))

            if args.log_print:
              print('# kernel_set:', self.kernel_set[i])
              print('# self.tconv[i](input):', self.tconv[i](input).shape)

        # 依照最後一層的feature dim(-x[-1].size(3)), 縮減各層的dim
        # 各層feature只取: [..., -x[-1].size(3): ]
        for i in range(len(self.kernel_set)):

            if args.log_print:
              print('# kernel_set:', self.kernel_set[i])
              print('# x[i].shape', x[i].shape)
              print('# -x[-1].size(3)', -x[-1].size(3))

            x[i] = x[i][...,-x[-1].size(3):]
            if args.log_print:
              print('# modeified x[i].shape', x[i].shape)

        # 8x4 kernel set => (64, 32, 207, 13)
        x = torch.cat(x,dim=1)
        if args.log_print:
          print("# final x", x.shape)
        #sys.exit()
        return x

# Paper's graph adjacency matrix
class graph_adaptive(nn.Module):
    # k: subgraph size
    def __init__(self, model_type, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_adaptive, self).__init__()

        self.model_type = model_type
        self.nnodes = nnodes
        '''
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
        '''

        if self.model_type == "GWNET":
            # Paper: "..initialize node embeddings by a uniform distribution with a size of 10."
            self.nodevec1 = nn.Parameter(torch.randn(self.nnodes, 10).to(device), requires_grad=True).to(device)
            self.nodevec2 = nn.Parameter(torch.randn(10, self.nnodes).to(device), requires_grad=True).to(device)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx=None):   # idx: 已打亂順序的index
        if self.model_type == "GWNET":
            adj = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)  # SoftMax(ReLU(E1*E2^T))

        return adj


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)