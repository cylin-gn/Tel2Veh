
##################

count = 0
for iter, ((x1, y1), (x2, y2)) in enumerate(zip(dataloader_gct['test_loader'].get_iterator(), dataloader_cctv['test_loader'].get_iterator())):

    count += y2.shape[0]
    print('y2', y2, y2.shape,count)

"""### Main(改output = CCTV) 有shuffle"""

# test_model(engine,"fusion",dataloader_gct,checkpoint,runid,dataloader_cctv)
def test_model(engine,model_type,dataloader,checkpoint,runid,dataloader2=None):

    ### 測試讀取出的model ###
    valid_loss = []
    valid_mape = []
    valid_rmse = []
    valid_smape = []
    s1 = time.time()

    if dataloader2 != None:
        print("in fusion.................")
        for iter, ((x1, y1), (x2, y2)) in enumerate(zip(dataloader['val_loader'].get_iterator(), dataloader2['val_loader'].get_iterator())):
            # Process data from first loader
            testx1 = torch.Tensor(x1).to(device)
            testx1 = testx1.transpose(1, 3)
            testy1 = torch.Tensor(y1).to(device)
            testy1 = testy1.transpose(1, 3)

            # Process data from second loader
            testx2 = torch.Tensor(x2).to(device)
            testx2 = testx2.transpose(1, 3)
            testy2 = torch.Tensor(y2).to(device)
            testy2 = testy2.transpose(1, 3)

            metrics = engine.eval('fusion',testx1, testy2[:,0,:,:], input_2= testx2)

            valid_loss_2 = []
            valid_mape_2 = []
            valid_rmse_2 = []
            valid_smape_2 = []

            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_smape.append(metrics[3])

            valid_loss_2.append(metrics[4])
            valid_mape_2.append(metrics[5])
            valid_rmse_2.append(metrics[6])
            valid_smape_2.append(metrics[7])
    else:
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(model_type,testx, testy[:,0,:,:])

            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_smape.append(metrics[3])

    mvalid_loss = np.mean(valid_loss)
    mvalid_mape = np.mean(valid_mape)
    mvalid_rmse = np.mean(valid_rmse)
    print("### 2-The valid loss on loding model is", str(round(mvalid_loss,4)))
    if dataloader2 != None:
      print("### 2-The valid(unseen) loss on loding model is", str(round(np.mean(valid_loss_2),4)))
    ### 測試讀取出的model ###


    #valid data
    outputs = []

    if dataloader2 != None:
      realy = torch.Tensor(dataloader2['y_val'+_types]).to(device)
      #realy2 = realy.transpose(1,3)[:,0,:5]
      #print('L59', realy2.shape, realy2[0,:,:2])

      filtered_indices = [index for index in args.all_idx if index not in args.exclue_idx]
      #print('L62', filtered_indices)
      selected_real = realy.transpose(1,3)[:,0,filtered_indices]
      #print('L64',selected_real.shape, selected_real[0,:,:2])
      #sys.exit()

      realy = selected_real
    else:
      realy = torch.Tensor(dataloader['y_val'+_types]).to(device)
      realy = realy.transpose(1,3)[:,0,:,:]
    print('#realy(valid)', realy.shape)

    '''
    for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():

            if model_type == 'gct':
              preds = engine.model(testx)
            elif model_type == 'cctv':
              preds = engine.model_2(testx)

            preds = preds.transpose(1,3)  # 64,1,6,12

        outputs.append(preds.squeeze()) # 64,1,6,12 ->squeeze()->64,6,12
    '''

    if dataloader2 != None:
        print("in fusion................. 2")
        for iter, ((x1, y1), (x2, y2)) in enumerate(zip(dataloader['val_loader'].get_iterator(), dataloader2['val_loader'].get_iterator())):
            # Process data from first loader
            testx1 = torch.Tensor(x1).to(device)
            testx1 = testx1.transpose(1, 3)

            # Process data from second loader
            testx2 = torch.Tensor(x2).to(device)
            testx2 = testx2.transpose(1, 3)

            with torch.no_grad():  #************#
                '''
                preds = engine.model_3(testx1, idx=None, input_2=testx2)

                preds = preds.transpose(1,3)  # 64,1,6,12
                '''
                # 要先經過model,model_2 encoder #
                output = engine.model(testx1) # cctv
                output = output.transpose(1,3)

                output_2 = engine.model_2(testx2) # cctv
                output_2 = output_2.transpose(1,3)

                output = torch.cat([output,testx1[:,1].unsqueeze(1)],dim=1)
                output_2 = torch.cat([output_2,testx2[:,1].unsqueeze(1)],dim=1)

                preds = engine.model_3(output, idx=None, input_2=output_2)
                preds = preds.transpose(1,3)  # 64,1,6,12


            outputs.append(preds.squeeze()) # 64,1,6,12 ->squeeze()->64,6,12
    else:
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1,3)
            with torch.no_grad():

                if model_type == 'gct':
                  preds = engine.model(testx)
                elif model_type == 'cctv':
                  preds = engine.model_2(testx)

                preds = preds.transpose(1,3)  # 64,1,6,12

            outputs.append(preds.squeeze()) # 64,1,6,12 ->squeeze()->64,6,12

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]  # 5240,6,12

    if dataloader2 != None:
      #indices = torch.tensor([7, 21, 19, 17, 3]).cuda() # The indices you want to select
      #print('L132', indices)

      filtered_corre_indices = [args.corre_idx[w] for w in range(len(args.corre_idx)) if w not in args.exclue_idx]
      indices = torch.tensor(filtered_corre_indices).cuda()
      #print('L136', indices)
      #sys.exit()

      yhat = torch.index_select(yhat, 1, indices)
    print('# cat valid preds', yhat.shape)


    pred = dataloader['scaler'].inverse_transform(yhat)

    vmae, vmape, vrmse,vsmape = metric(pred,realy)
    print("valid vmae",vmae)


    #test data
    outputs = []
    realy = torch.Tensor(dataloader['y_test'+_types]).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]


    if dataloader2 != None:
      realy = torch.Tensor(dataloader2['y_test'+_types]).to(device)

      filtered_indices = [index for index in args.all_idx if index not in args.exclue_idx]
      selected_real = realy.transpose(1,3)[:,0,filtered_indices]

      realy_seen = selected_real
      #-----------#
      filtered_indices = [index for index in args.all_idx if index in args.exclue_idx]
      selected_real = realy.transpose(1,3)[:,0,filtered_indices]

      realy_unseen = selected_real
    else:
      realy = torch.Tensor(dataloader['y_test'+_types]).to(device)
      realy = realy.transpose(1,3)[:,0,:,:]
    print('#realy(test)', realy.shape)

    if dataloader2 != None:
        for iter, ((x1, y1), (x2, y2)) in enumerate(zip(dataloader['test_loader'].get_iterator(), dataloader2['test_loader'].get_iterator())):
            # Process data from first loader
            testx1 = torch.Tensor(x1).to(device)
            testx1 = testx1.transpose(1, 3)

            # Process data from second loader
            testx2 = torch.Tensor(x2).to(device)
            testx2 = testx2.transpose(1, 3)
            '''
            with torch.no_grad():
                preds = engine.model_3(testx1, idx=None, input_2=testx2)

                preds = preds.transpose(1,3)  # 64,1,6,12
            '''
            with torch.no_grad():  #************#
                '''
                preds = engine.model_3(testx1, idx=None, input_2=testx2)

                preds = preds.transpose(1,3)  # 64,1,6,12
                '''
                # 要先經過model,model_2 encoder #
                output = engine.model(testx1) # cctv
                output = output.transpose(1,3)

                output_2 = engine.model_2(testx2) # cctv
                output_2 = output_2.transpose(1,3)

                output = torch.cat([output,testx1[:,1].unsqueeze(1)],dim=1)
                output_2 = torch.cat([output_2,testx2[:,1].unsqueeze(1)],dim=1)

                preds = engine.model_3(output, idx=None, input_2=output_2)
                preds = preds.transpose(1,3)  # 64,1,6,12

            outputs.append(preds.squeeze()) # 64,1,6,12 ->squeeze()->64,6,12
    else:
        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            with torch.no_grad():
                #preds = engine.model(testx)
                if model_type == 'gct':
                  preds = engine.model(testx)
                elif model_type == 'cctv':
                  preds = engine.model_2(testx)

                preds = preds.transpose(1, 3)
            outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]  #10478, 6, 12

    if dataloader2 != None:
      #indices = torch.tensor([7, 21, 19, 17, 3]).cuda() # The indices you want to select
      #print('L218', indices)

      filtered_corre_indices = [args.corre_idx[w] for w in range(len(args.corre_idx)) if w not in args.exclue_idx]
      indices = torch.tensor(filtered_corre_indices).cuda()
      #print('L222', indices)
      #sys.exit()

      #print('L243', yhat.shape)
      yhat_seen = torch.index_select(yhat, 1, indices)
      #print('L245', indices, yhat.shape)
      #------------------#
      filtered_corre_indices = [args.corre_idx[w] for w in range(len(args.corre_idx)) if w in args.exclue_idx]
      indices = torch.tensor(filtered_corre_indices).cuda()

      yhat_unseen = torch.index_select(yhat, 1, indices)


    if dataloader2 != None:
      #------------unseen------------#
      print("#-------------------unseen--------------------#")
      mae = []
      mape = []
      rmse = []
      smape = []
      for i in range(args.seq_out_len):

          pred = dataloader['scaler'].inverse_transform(yhat_unseen[:, :, i])

          real = realy_unseen[:, :, i]
          metrics = metric(pred, real)

          #print(metrics, 'pred', pred.shape , pred[0], 'real', real.shape, real[0])

          log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
          print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
          mae.append(metrics[0])
          mape.append(metrics[1])
          rmse.append(metrics[2])
          smape.append(metrics[3])

          if i==2 or i==5 or i == 11:
              ground_truth = real.cpu()
              predictions = pred.cpu()

              num_samples = ground_truth.size(0)
              nodes = ground_truth.shape[1]
              time_steps_per_chart = 157
              num_charts = num_samples // time_steps_per_chart
              #if num_samples % time_steps_per_chart != 0:
              #    num_charts += 1

              for node in range(nodes):
                for chart_num in range(num_charts):
                    if chart_num == 0:
                        gt_segment = ground_truth[num_samples-(157-(11-i)):num_samples, node]
                        pred_segment = predictions[num_samples-(157-(11-i)):num_samples, node]
                    else:
                        gt_segment = ground_truth[(num_samples-(157-(11-i)))-157*(chart_num):(num_samples-(157-(11-i)))-157*(chart_num-1), node]
                        pred_segment = predictions[(num_samples-(157-(11-i)))-157*(chart_num):(num_samples-(157-(11-i)))-157*(chart_num-1), node]


                    # Reshape for plotting
                    gt_segment = gt_segment.view(-1).numpy()
                    pred_segment = pred_segment.view(-1).numpy()

                    # Create the plot
                    plt.figure(figsize=(10, 4))
                    plt.plot(gt_segment, label='Ground Truth')
                    plt.plot(pred_segment, label='Prediction')
                    plt.title(f'Prediction len {i+1}, Day 9/{26-chart_num}, MAE:{round(metrics[0], 2)}, MAPE:{round(metrics[1], 2)}')
                    plt.xlabel('Time Step')
                    plt.ylabel('Value')
                    plt.legend()
                    plt.show()

                    print("#real", gt_segment)
                    print("#predict", pred_segment)

      log = '{:.2f}	{:.2f}	{:.4f}	{:.4f}	'
      print( "##### exp" + str(args.expid_gct) + "_" + str(runid)+'	',
            log.format(mae[0], rmse[0], smape[0], mape[0]),
            log.format(mae[2], rmse[2], smape[2], mape[2]),
            log.format(mae[5], rmse[5], smape[5], mape[5]),
            log.format(mae[11], rmse[11], smape[11], mape[11]),
          )

      #------------seen------------#
      print("#-------------------seen--------------------#")
      mae = []
      mape = []
      rmse = []
      smape = []
      for i in range(args.seq_out_len):

          pred = dataloader['scaler'].inverse_transform(yhat_seen[:, :, i])

          real = realy_seen[:, :, i]
          metrics = metric(pred, real)

          #print(metrics, 'pred', pred.shape , pred[0], 'real', real.shape, real[0])

          log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
          print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
          mae.append(metrics[0])
          mape.append(metrics[1])
          rmse.append(metrics[2])
          smape.append(metrics[3])

          if i == 11:
              ground_truth = real.cpu()
              predictions = pred.cpu()

              num_samples = ground_truth.size(0)
              nodes = ground_truth.shape[1]
              time_steps_per_chart = 157
              num_charts = num_samples // time_steps_per_chart
              if num_samples % time_steps_per_chart != 0:
                  num_charts += 1

              for node in range(nodes):
                for chart_num in range(num_charts):
                    if chart_num == 0:
                        gt_segment = ground_truth[num_samples-157:, node]
                        pred_segment = predictions[num_samples-157:, node]
                        #print(gt_segment,len(gt_segment))
                        print('last gt_segment', gt_segment)
                    else:
                        gt_segment = ground_truth[num_samples-157*(chart_num+1):num_samples-157*(chart_num), node]
                        pred_segment = predictions[num_samples-157*(chart_num+1):num_samples-157*(chart_num), node]
                        #print(gt_segment,len(gt_segment))


                    # Reshape for plotting
                    gt_segment = gt_segment.view(-1).numpy()
                    pred_segment = pred_segment.view(-1).numpy()

                    # Create the plot
                    plt.figure(figsize=(10, 4))
                    plt.plot(gt_segment, label='Ground Truth')
                    plt.plot(pred_segment, label='Prediction')
                    plt.title(f'Node {node+1}, Segment {chart_num+1}')
                    plt.xlabel('Time Step')
                    plt.ylabel('Value')
                    plt.legend()
                    plt.show()

      log = '{:.2f}	{:.2f}	{:.4f}	{:.4f}	'
      print( "##### exp" + str(args.expid_gct) + "_" + str(runid)+'	',
            log.format(mae[0], rmse[0], smape[0], mape[0]),
            log.format(mae[2], rmse[2], smape[2], mape[2]),
            log.format(mae[5], rmse[5], smape[5], mape[5]),
            log.format(mae[11], rmse[11], smape[11], mape[11]),
          )
    else:
      mae = []
      mape = []
      rmse = []
      smape = []
      for i in range(args.seq_out_len):

          pred = dataloader['scaler'].inverse_transform(yhat[:, :, i])

          real = realy[:, :, i]
          metrics = metric(pred, real)

          #print(metrics, 'pred', pred.shape , pred[0], 'real', real.shape, real[0])

          log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
          print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
          mae.append(metrics[0])
          mape.append(metrics[1])
          rmse.append(metrics[2])
          smape.append(metrics[3])

      log = '{:.2f}	{:.2f}	{:.4f}	{:.4f}	'
      print( "##### exp" + str(args.expid_gct) + "_" + str(runid)+'	',
            log.format(mae[0], rmse[0], smape[0], mape[0]),
            log.format(mae[2], rmse[2], smape[2], mape[2]),
            log.format(mae[5], rmse[5], smape[5], mape[5]),
            log.format(mae[11], rmse[11], smape[11], mape[11]),
          )

    ### Drawing Loss Diagram ###
    fig = plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(checkpoint['train_loss'], label="train loss")
    plt.plot(checkpoint['valid_loss'], label="valid loss")
    plt.legend(loc="upper right")
    plt.title('#Loss of Training', fontsize=20)
    plt.ylabel("MAPE", fontsize=14)
    plt.xlabel("Epochs", fontsize=14)
    plt.show()

    return vmae, vmape, vrmse,vsmape, mae, mape, rmse,smape

def main(runid):


    # if args.load_static_feature:
    #     static_feat = load_node_feature('data/sensor_graph/location.csv')
    # else:
    #     static_feat = None

    # num_nodes, subgraph_size, adj_mx
    model = gginet(args.model_type, args.gcn_true, args.buildA_true, args.gcn_depth,
                   args.num_nodes_gct, #********#
                   device,
                   predefined_A=adj_mx_gct,
                   kernel_set=args.kernel_set, dropout=args.dropout,
                   subgraph_size=args.subgraph_size_gct,  #********#
                   node_dim=args.node_dim, dilation_exponential=args.dilation_exponential, conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True)

    model_cctv = gginet(args.model_type, args.gcn_true, args.buildA_true, args.gcn_depth,
                   args.num_nodes_cctv, #********#
                   device,
                   predefined_A=adj_mx_cctv,
                   kernel_set=args.kernel_set, dropout=args.dropout,
                   subgraph_size=args.subgraph_size_cctv, #********#
                   node_dim=args.node_dim, dilation_exponential=args.dilation_exponential, conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True)

    model_fusion = gginet(args.model_type, args.gcn_true, args.buildA_true, args.gcn_depth,
                   args.num_nodes_gct, #********#
                   device,
                   predefined_A=adj_mx_gct,
                   kernel_set=args.kernel_set, dropout=args.dropout,
                   subgraph_size=args.subgraph_size_gct,  #********#
                   node_dim=args.node_dim, dilation_exponential=args.dilation_exponential, conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True, fusion=1)

    print(model)
    print(args)

    print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])       # model參數量!
    print('Number of model parameters is', nParams)

    engine = Trainer(model, model_cctv,model_fusion, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, dataloader_gct['scaler'], dataloader_cctv['scaler'], device, args.cl)

    ###############
    #.............#
    ###############

    # expid_gct
    SAVE_PATH = args.save + "exp" + str(args.expid_gct) + "_" + str(runid) +".pth"
    print("### loading model is:",SAVE_PATH ,'###')
    checkpoint = torch.load(SAVE_PATH)
    engine.model.load_state_dict(checkpoint['model_state_dict'])
    engine.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    print("### Loading Model finished ###")
    print("### The valid loss on loding model is", str(round(loss,4)))

    # expid_cctv
    SAVE_PATH = args.save + "exp" + str(args.expid_cctv) + "_" + str(runid) +".pth"
    print("### loading model is:",SAVE_PATH ,'###')
    checkpoint = torch.load(SAVE_PATH)
    engine.model_2.load_state_dict(checkpoint['model_state_dict'])
    engine.optimizer_2.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    print("### Loading Model finished ###")
    print("### The valid loss on loding model is", str(round(loss,4)))

    #-----------------------Training--------------------#

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    minl = 1e5
    start_epoch=0
    SAVE_PATH = ""
    train_loss_epoch = []  # 紀錄train在epoch收斂
    valid_loss_epoch = []  # 紀錄valid在epoch收斂

    for i in range(start_epoch,start_epoch+args.epochs+1):

        train_loss_gct = []
        train_loss = []
        train_mape = []
        train_rmse = []
        train_smape = []

        ### unseen ###
        train_loss_2 = []
        train_mape_2 = []
        train_rmse_2 = []
        train_smape_2 = []

        t1 = time.time()

        permutation = np.random.permutation(dataloader_gct['train_loader'].size)
        dataloader_gct['train_loader'].set_permutation(permutation)
        dataloader_cctv['train_loader'].set_permutation(permutation)

        for iter, ((x1, y1), (x2, y2)) in enumerate(zip(dataloader_gct['train_loader'].get_iterator(), dataloader_cctv['train_loader'].get_iterator())):
            # Process data from first loader (GCT)
            trainx1 = torch.Tensor(x1).to(device)
            trainx1 = trainx1.transpose(1, 3)
            trainy1 = torch.Tensor(y1).to(device)
            trainy1 = trainy1.transpose(1, 3)

            # Process data from second loader (CCTV)
            trainx2 = torch.Tensor(x2).to(device)
            trainx2 = trainx2.transpose(1, 3)
            trainy2 = torch.Tensor(y2).to(device)
            trainy2 = trainy2.transpose(1, 3)

            # 目標仍是predict GCT
            metrics = engine.train(trainx1,trainx2, trainy1[:,0,:,:], trainy2[:,0,:,:])

            train_loss_gct.append(metrics[0])
            train_loss.append(metrics[1])
            train_mape.append(metrics[2])
            train_rmse.append(metrics[3])
            train_smape.append(metrics[4])

            train_loss_2.append(metrics[5])
            train_mape_2.append(metrics[6])
            train_rmse_2.append(metrics[7])
            train_smape_2.append(metrics[8])

            #if iter % args.print_every == 0 :
            #    log = 'Train Iter: {:03d}, [Seen] MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}, [Unseen] MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'
            #    print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1], train_loss_2[-1], train_mape_2[-1], train_smape_2[-1]),flush=True)

        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        valid_smape = []

        valid_loss_2 = []
        valid_mape_2 = []
        valid_rmse_2 = []
        valid_smape_2 = []

        s1 = time.time()


        for iter, ((x1, y1), (x2, y2)) in enumerate(zip(dataloader_gct['val_loader'].get_iterator(), dataloader_cctv['val_loader'].get_iterator())):
            # Process data from first loader (GCT)
            testx1 = torch.Tensor(x1).to(device)
            testx1 = testx1.transpose(1, 3)
            testy1 = torch.Tensor(y1).to(device)
            testy1 = testy1.transpose(1, 3)

            # Process data from second loader (CCTV)
            testx2 = torch.Tensor(x2).to(device)
            testx2 = testx2.transpose(1, 3)
            testy2 = torch.Tensor(y2).to(device)
            testy2 = testy2.transpose(1, 3)

            metrics = engine.eval('fusion',testx1, testy2[:,0,:,:], input_2= testx2)

            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_smape.append(metrics[3])

            valid_loss_2.append(metrics[4])
            valid_mape_2.append(metrics[5])
            valid_rmse_2.append(metrics[6])
            valid_smape_2.append(metrics[7])

        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)


        mtrain_loss_gct = np.mean(train_loss_gct)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_smape = np.mean(train_smape)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_smape = np.mean(valid_smape)
        his_loss.append(mtrain_loss)
        #-------------#
        mtrain_loss_2 = np.mean(train_loss_2)
        mtrain_mape_2 = np.mean(train_mape_2)
        mtrain_rmse_2 = np.mean(train_rmse_2)
        mtrain_smape_2 = np.mean(train_smape_2)

        mvalid_loss_2 = np.mean(valid_loss_2)
        mvalid_mape_2 = np.mean(valid_mape_2)
        mvalid_rmse_2 = np.mean(valid_rmse_2)
        mvalid_smape_2 = np.mean(valid_smape_2)

        w1_value = model_fusion.W_f1.item()
        w2_value = model_fusion.W_f2.item()

        log = 'Training Epoch: {:03d}, [Seen] MAE(GCT): {:.4f}, MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f} || Valid MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'
        print(log.format(i, mtrain_loss_gct, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse ),flush=True)

        log = 'Training Epoch: {:03d}, [Unseen] MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f} || Valid MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'
        print(log.format(i, mtrain_loss_2, mtrain_mape_2, mtrain_rmse_2, mvalid_loss_2, mvalid_mape_2, mvalid_rmse_2),flush=True)

        log = 'Training w1: {:.7f}, w2: {:.7f}, Training Time: {:.4f}/epoch'
        print(log.format(w1_value, w2_value, (t2 - t1)),flush=True)

        # 紀錄每個epoch的loss
        train_loss_epoch.append(mtrain_loss)
        valid_loss_epoch.append(mvalid_loss)

        if mvalid_loss<minl:
            #torch.save(engine.model.state_dict(), args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth")
            SAVE_PATH = args.save + "exp" + str(args.expid_fusion) + "_" + str(runid) +".pth"
            print("### Update Best Model:",SAVE_PATH, " ###")
            torch.save({
              'epoch': i,
              'task_level': engine.task_level,
              'model_state_dict': engine.model_3.state_dict(),   #*******#
              'optimizer_state_dict': engine.optimizer_3.state_dict(), #*******#
              'loss': mvalid_loss,
              'train_loss': train_loss_epoch,
              'valid_loss': valid_loss_epoch
            }, SAVE_PATH)
            minl = mvalid_loss

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


    bestid = np.argmin(his_loss)

    writer.close()
    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))

    #-----------------------Training-------------------#

    print("--------------------CCTV------------------")
    vmae, vmape, vrmse,vsmape, mae, mape, rmse,smape = test_model(engine,"cctv",dataloader_cctv,checkpoint,runid)
    print("--------------------GCT------------------")
    vmae, vmape, vrmse,vsmape, mae, mape, rmse,smape = test_model(engine,"gct",dataloader_gct,checkpoint,runid)
    print("--------------------Fusion------------------")
    # expid_cctv
    SAVE_PATH = args.save + "exp" + str(args.expid_fusion) + "_" + str(runid) +".pth"
    print("### loading model is:",SAVE_PATH ,'###')
    checkpoint = torch.load(SAVE_PATH)
    engine.model_3.load_state_dict(checkpoint['model_state_dict'])   #*******#
    engine.optimizer_3.load_state_dict(checkpoint['optimizer_state_dict']) #*******#
    loss = checkpoint['loss']
    print("### Loading Model finished ###")
    print("### The valid loss on loding model is", str(round(loss,4)))
    vmae, vmape, vrmse,vsmape, mae, mape, rmse,smape = test_model(engine,"fusion",dataloader_gct,checkpoint,runid,dataloader_cctv)

    #sys.exit()

    return vmae, vmape, vrmse,vsmape, mae, mape, rmse,smape


if __name__ == "__main__":

    vmae = []
    vmape = []
    vrmse = []
    vsmape = []
    mae = []
    mape = []
    rmse = []
    smape = []
    for i in range(args.runs):
        vm1, vm2, vm3,vm4, m1, m2, m3, m4 = main(i)
        vmae.append(vm1)
        vmape.append(vm2)
        vrmse.append(vm3)
        vsmape.append(vm4)
        mae.append(m1)
        mape.append(m2)
        rmse.append(m3)
        smape.append(m4)

    mae = np.array(mae)
    mape = np.array(mape)
    rmse = np.array(rmse)
    smape = np.array(smape)

    amae = np.mean(mae,0)
    amape = np.mean(mape,0)
    armse = np.mean(rmse,0)
    asmape = np.mean(smape,0)

    smae = np.std(mae,0)
    s_mape = np.std(mape,0)
    srmse = np.std(rmse,0)
    s_smape = np.std(smape,0)

    print('\n\nResults for 10 runs\n\n')
    #valid data
    print('valid\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(vmae),np.mean(vrmse),np.mean(vmape)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(vmae),np.std(vrmse),np.std(vmape)))
    print('\n\n')
    #test data
    print('test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std')
    for i in [2,5,11]:
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
        print(log.format(i+1, amae[i], armse[i], amape[i], smae[i], srmse[i], s_mape[i]))