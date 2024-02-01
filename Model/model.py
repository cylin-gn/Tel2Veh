"""### Model .00001 ~ .0001"""

class gginet(nn.Module):
    def __init__(self, model_type, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None,kernel_set=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True, fusion=None):
        super(gginet, self).__init__()

        self.model_type = model_type

        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.layers = layers
        self.seq_length = seq_length

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))


        if fusion != None:  #********#

          self.fusion_list = nn.ModuleList()
          in_channel = 32
          n_heads = 8
          dropout = 0
          alpha = 0.2
          t_len = 13

          kern = 2
          dilation_factor = 1
          n_heads = 8
          target_len = t_len
          self.fusion_list.append(S_MutiChannel_GAT(kern, dilation_factor, n_heads, target_len, [24,16,8], [16,24,32], dropout))




        # 一整個gtnet只會有一組node embedding E1,E2
        self.adaptive_mx = graph_adaptive(self.model_type, num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)


        if self.model_type == "GWNET":
            kernel_size = 2
            self.receptive_field = int((layers/2)* ( (kernel_size-1)+ kernel_size ) + 1)    # kernel: 1,2,1,2...

        print("# Model Type", self.model_type)
        print("# receptive_field", self.receptive_field)
        i=0
        if dilation_exponential>1:
            rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            rf_size_i = i*layers*(kernel_size-1)+1
        new_dilation = 1
        for j in range(1,layers+1):

            if self.model_type == "GWNET":
                if j % 2 == 1:
                    new_dilation = 1
                elif j % 2 == 0:
                    new_dilation = 2

            # residual_channels: 32, conv_channels: 32 , new_dilation: 1
            self.filter_convs.append(dilated_inception(kernel_set, residual_channels, conv_channels, dilation_factor=new_dilation))
            self.gate_convs.append(dilated_inception(kernel_set, residual_channels, conv_channels, dilation_factor=new_dilation))


            # 1x1 convolution for skip connection
            #(0): Conv1d(32, 256, kernel_size=(1, 1), stride=(1,))
            #(1): Conv1d(32, 256, kernel_size=(1, 1), stride=(1,))
            #(2): Conv1d(32, 256, kernel_size=(1, 1), stride=(1,))
            #...
            if self.model_type == "GWNET" :
                self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                        out_channels=skip_channels,
                                        kernel_size=(1, 1)))

            #####   GCN   ##### START
            if self.gcn_true:
                if self.model_type == "GWNET":
                    if self.buildA_true:    # 有加入adaptive matrix
                        adj_mx_len = 3
                    else:
                        adj_mx_len = 2
                    self.gconv1.append(gcn(self.model_type, adj_mx_len, conv_channels, residual_channels, gcn_depth, dropout, propalpha))


            else:
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                out_channels=residual_channels,
                                              kernel_size=(1, 1)))

            #####   GCN   ##### END

            #####   Normalization   ##### START
            if self.model_type == "GWNET":
                self.norm.append(nn.BatchNorm2d(residual_channels))
            #####   Normalization   ##### END

            new_dilation *= dilation_exponential



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)



        self.idx = torch.arange(self.num_nodes).to(device)

        if fusion != None:  #********#
          self.W_f1 = nn.Parameter(torch.FloatTensor(1, 1).uniform_(0.00001, 0.0001))
          self.W_f2 = nn.Parameter(torch.FloatTensor(1, 1).uniform_(0.00001, 0.0001))



    def forward(self, input, idx=None, input_2=None):

        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        # Step0: 檢查receptive_field, 不足則padding0
        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))

            if input_2 != None:   #********#
              input_2 = nn.functional.pad(input_2,(self.receptive_field-self.seq_length,0,0,0))

        # Step0: 利用node idx建立embedding:E1, E2
        # 建立Sparse的Adaptive matrix
        if self.gcn_true:

            # Use adaptive matrix
            if self.buildA_true:
                if self.model_type == "GWNET":
                    adp = self.adaptive_mx()
                    adp = self.predefined_A + [adp]


            else:
                adp = self.predefined_A

        if self.model_type == "GWNET":
            skip = 0

        if input_2 != None:  #********#
            x_all = []
            corre_idx_selected = [args.corre_idx[w] for w in range(len(args.corre_idx)) if w not in args.exclue_idx]
            all_idx_selected = [index for index in args.all_idx if index not in args.exclue_idx]

            _count = 0
            for i in range(49):
              if i in corre_idx_selected:
                x_all.append(input_2[:,:, all_idx_selected[_count]].unsqueeze(2))
                _count+=1
              else:
                x_all.append(input[:,:,i].unsqueeze(2))
            x = torch.cat(x_all,2)

            x = self.start_conv(x)
        else:
          # Step1: turn([64, 2, 207, 19]) to ([64, 32, 207, 19])
          x = self.start_conv(input)      # gct


        if input_2 != None:   #********#
            x2 = self.start_conv(input_2)   # cctv
            x = self.fusion_list[0](x,x2,adj_mx_gct[0])


        # Layers : 3層 : 19->13->7->1 (取決於TCN取的維度)
        for i in range(self.layers):

            # Step2: Temporal Model --START #
            # 為上一層輸出, ex:  [64, 32, 207, 19] -> [64, 32, 207, 13] -> [64, 32, 207, 7]-> [64, 32, 207, 1]
            residual = x

            # Tanh
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)

            # Sigmoid
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)

            # Fusion
            x = filter * gate

            # Step2: Temporal Model --END #

            # Step3: Skip after TCN --START #
            s = x
            '''
            # skip_convs #
            (0): Conv2d(32, 64, kernel_size=(1, 13), stride=(1, 1))
            (1): Conv2d(32, 64, kernel_size=(1, 7), stride=(1, 1))
            (2): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
            '''
            # fusion output:([64, 32, 207, 13])
            # skip_convsL 0:([64, 64, 207, 1])
            s = self.skip_convs[i](s)

            if self.model_type == "GWNET":
                # 讓上一個skip配合目前的skip
                try:
                    skip = skip[:, :, :,  -s.size(3):]
                except:
                    skip = 0

            skip = s + skip

            # Step3: Skip after TCN --END #


            # Step4: GCN --START #
            if self.gcn_true:
                if self.model_type == "GWNET":
                    x = self.gconv1[i](x, adp)

            else:
                x = self.residual_convs[i](x)

            # x 經過dilated處理後, 會減少feature維度, ex: 19->13->7->1
            # 而residual為上一層輸出, 維度為: 19, 13 ...
            # 所以需要配合x進行維度調整: [:, :, :, -x.size(3):], 然後進行elemenet-wise相加
            x = x + residual[:, :, :, -x.size(3):]


            if self.model_type == "GWNET":
                x = self.norm[i](x)

            # Step4: GCN --END #


        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x