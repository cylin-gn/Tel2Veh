"""### Trainer  """

class Trainer():
    def __init__(self, model,model_2, model_3, lrate, wdecay, clip, step_size, seq_out_len, scaler, scaler_2, device, cl=True):

        # GCT
        self.scaler = scaler
        self.model = model
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)

        # CCTV
        self.scaler_2 = scaler_2
        self.model_2 = model_2
        self.model_2.to(device)
        self.optimizer_2 = optim.Adam(self.model_2.parameters(), lr=lrate, weight_decay=wdecay)

        # Fusion
        self.model_3 = model_3
        self.model_3.to(device)
        self.optimizer_3 = optim.Adam(self.model_3.parameters(), lr=lrate, weight_decay=wdecay)

        self.loss = masked_mae
        self.clip = clip
        self.step = step_size
        self.iter = 1
        self.task_level = 1
        self.seq_out_len = seq_out_len
        self.cl = cl



    # input: GCT, input_2: CCTV, real_val: gct
    def train(self, input,input_2, real_val, real_val_2, idx=None):
        # torch.Size([64, 2, 49, 12]) torch.Size([64, 2, 6, 12]) torch.Size([64, 49, 12]) torch.Size([64, 6, 12])
        #print('input,input_2, real_val, real_val_2', input.shape,input_2.shape, real_val.shape, real_val_2.shape)  # torch.Size([64, 2, 49, 12]) torch.Size([64, 2, 6, 12])


        self.model.eval()
        self.model_2.eval()

        output = self.model(input)
        output = output.transpose(1,3)
        output_2 = self.model_2(input_2)
        output_2 = output_2.transpose(1,3)

        output = torch.cat([output,input[:,1].unsqueeze(1)],dim=1)
        output_2 = torch.cat([output_2,input_2[:,1].unsqueeze(1)],dim=1)

        self.model_3.train()  #*******#
        self.optimizer_3.zero_grad()  #*******#
        output = self.model_3(output, idx=idx, input_2=output_2) #*******#
        output = output.transpose(1,3)  # torch.Size([64, 2, 49, 12])

        real_1 = torch.unsqueeze(real_val,dim=1)   # torch.Size([64, 49, 12])
        real_2 = torch.unsqueeze(real_val_2,dim=1) # torch.Size([64, 6, 12])

        predict = self.scaler.inverse_transform(output) # torch.Size([64, 49, 12])

        #------------------主要預測點位逼近影像-----------------#
        filtered_corre_indices = [args.corre_idx[w] for w in range(len(args.corre_idx)) if w not in args.exclue_idx]
        #indices = torch.tensor(filtered_corre_indices).cuda()
        #selected_predict = torch.index_select(predict, 2, indices)
        selected_predict = predict[:,:,filtered_corre_indices]

        filtered_indices = [index for index in args.all_idx if index not in args.exclue_idx]
        selected_real = real_2[:,:,filtered_indices]

        predict1 = selected_predict
        real1 = selected_real

        if self.iter%self.step==0 and self.task_level<=self.seq_out_len:
            self.task_level +=1
            print("### cl learning\n iter",self.iter,"\niter%step",self.iter%self.step,"\ntask_level",self.task_level)
            print("# predict1(GCT->video) len:", len(predict1[:, :, :, :self.task_level]))
            #print(" ####### dynamic fusion weight", self.model_3.W_f1, self.model_3.W_f2)


        if self.cl:
            loss1 = masked_mae(predict1[:, :, :, :self.task_level], real1[:, :, :, :self.task_level], 0.0)
            loss_mape_1 = masked_mape(predict1[:, :, :, :self.task_level], real1[:, :, :, :self.task_level], 0.0)
            loss_rmse_1 = masked_rmse(predict1[:, :, :, :self.task_level], real1[:, :, :, :self.task_level], 0.0)
            loss_smape_1 = masked_smape(predict1[:, :, :, :self.task_level], real1[:, :, :, :self.task_level], 0.0)

        else:
            loss1 = masked_mae(predict1, real1, 0.0)

            loss_mape_1 =  masked_mape(predict1, real1, 0.0)
            loss_rmse_1 =  masked_rmse(predict1, real1, 0.0)
            loss_smape_1 =  masked_smape(predict1, real1, 0.0)



        #------------------其他逼近GCT-----------------#
        numbers_0_to_48 = list(range(49))
        excluded_numbers_args = args.corre_idx
        not_seen = [num for num in numbers_0_to_48 if num not in excluded_numbers_args]

        indices = torch.tensor(not_seen).cuda()

        selected_predict = predict[:,:,indices]
        selected_real = real_1[:,:,not_seen]

        predict1_1 = selected_predict
        real1_1 = selected_real

        if self.cl:
            loss1_1 = masked_mae(predict1_1[:, :, :, :self.task_level], real1_1[:, :, :, :self.task_level], 0.0)

        else:
            loss1_1 = masked_mae(predict1_1, real1_1, 0.0)


        #------------------main unseen GCT-----------------#
        not_seen = args.exclue_idx

        indices = torch.tensor(not_seen).cuda()

        selected_predict = predict[:,:,indices]
        selected_real = real_1[:,:,not_seen]

        predict1_2 = selected_predict
        real1_2 = selected_real

        if self.cl:
            loss1_2 = masked_mae(predict1_2[:, :, :, :self.task_level], real1_1[:, :, :, :self.task_level], 0.0)

        else:
            loss1_2 = masked_mae(predict1_2, real1_2, 0.0)

        #------------------unseen點位誤差-----------------#

        filtered_corre_indices = [args.corre_idx[w] for w in range(len(args.corre_idx)) if w  in args.exclue_idx]
        #indices = torch.tensor(filtered_corre_indices).cuda()
        #selected_predict = torch.index_select(predict, 2, indices)
        selected_predict = predict[:,:,filtered_corre_indices]

        filtered_indices = [index for index in args.all_idx if index in args.exclue_idx]
        selected_real = real_2[:,:,filtered_indices]


        predict2 = selected_predict
        real2 = selected_real

        if self.cl:
            loss2 = masked_mae(predict2[:, :, :, :self.task_level], real2[:, :, :, :self.task_level], 0.0)

            loss_mape_2 = masked_mape(predict2[:, :, :, :self.task_level], real2[:, :, :, :self.task_level], 0.0)
            loss_rmse_2 = masked_rmse(predict2[:, :, :, :self.task_level], real2[:, :, :, :self.task_level], 0.0)
            loss_smape_2 = masked_smape(predict2[:, :, :, :self.task_level], real2[:, :, :, :self.task_level], 0.0)

        else:
            loss2 = masked_mae(predict2, real2, 0.0)

            loss_mape_2 =  masked_mape(predict2, real2, 0.0)
            loss_rmse_2 =  masked_rmse(predict2, real2, 0.0)
            loss_smape_2 =  masked_smape(predict2, real2, 0.0)

        var_w = (abs(self.model_3.W_f1)).to(loss1_1.device)
        var_w1 = (abs(self.model_3.W_f2)).to(loss1_2.device)

        loss = 1*loss1 + var_w*(loss1_1) + var_w1*(loss1_2)
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model_3.parameters(), self.clip)  #*******#

        self.optimizer_3.step() #*******#

        self.iter += 1
        return (var_w1*loss1_2.item()+var_w*loss1_1.item()).cpu().detach().numpy(), loss1.item(),loss_mape_1.item(),loss_rmse_1.item(),loss_smape_1.item(), loss2.item(),loss_mape_2.item(),loss_rmse_2.item(),loss_smape_2.item()

    def eval(self, model_type, input, real_val, input_2=None):
        #print('type',model_type)
        if model_type == 'gct':
          self.model.eval()
          output = self.model(input)
          output = output.transpose(1,3)

          predict = self.scaler.inverse_transform(output)

        elif model_type == 'cctv':
          self.model_2.eval()
          output = self.model_2(input)
          output = output.transpose(1,3)

          predict = self.scaler_2.inverse_transform(output)

        elif model_type == 'fusion':
          self.model.eval()
          self.model_2.eval()
          self.model_3.eval()

          output = self.model(input)
          output = output.transpose(1,3)
          output_2 = self.model_2(input_2)
          output_2 = output_2.transpose(1,3)

          output = torch.cat([output,input[:,1].unsqueeze(1)],dim=1)
          output_2 = torch.cat([output_2,input_2[:,1].unsqueeze(1)],dim=1)

          output = self.model_3(output, idx=None, input_2=output_2)
          output = output.transpose(1,3)

          predict = self.scaler.inverse_transform(output)

          # seen 誤差 #
          filtered_corre_indices = [args.corre_idx[w] for w in range(len(args.corre_idx)) if w not in args.exclue_idx]
          indices = torch.tensor(filtered_corre_indices).cuda()

          selected_predict1 = torch.index_select(predict, 2, indices)

          # unseen 誤差 #
          filtered_corre_indices = [args.corre_idx[w] for w in range(len(args.corre_idx)) if w  in args.exclue_idx]
          indices = torch.tensor(filtered_corre_indices).cuda()
          selected_predict2 = torch.index_select(predict, 2, indices)

        real = torch.unsqueeze(real_val,dim=1)


        if model_type == 'fusion':

          predict1 = selected_predict1
          predict2 = selected_predict2

          filtered_indices = [index for index in args.all_idx if index not in args.exclue_idx]
          selected_real = real[:,:,filtered_indices]
          real1 = selected_real

          filtered_indices = [index for index in args.all_idx if index in args.exclue_idx]
          selected_real = real[:,:,filtered_indices]
          real2 = selected_real

        if model_type != 'fusion':
          loss = self.loss(predict, real, 0.0)
          mape = masked_mape(predict,real,0.0).item()
          rmse = masked_rmse(predict,real,0.0).item()
          smape = masked_smape(predict,real,0.0).item()
          return loss.item(),mape,rmse,smape
        else:

          loss1 = self.loss(predict1, real1, 0.0)
          loss_mape_1 = masked_mape(predict1,real1,0.0)
          loss_rmse_1 = masked_rmse(predict1,real1,0.0)
          loss_smape_1 = masked_smape(predict1,real1,0.0)

          loss2 = self.loss(predict2, real2, 0.0)
          loss_mape_2 = masked_mape(predict2,real2,0.0)
          loss_rmse_2 = masked_rmse(predict2,real2,0.0)
          loss_smape_2 = masked_smape(predict2,real2,0.0)

          return loss1.item(),loss_mape_1.item(),loss_rmse_1.item(),loss_smape_1.item(), loss2.item(),loss_mape_2.item(),loss_rmse_2.item(),loss_smape_2.item()
