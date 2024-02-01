
"""### Loading Data CCTV 用gct scaler"""

batch_size = args.batch_size
valid_batch_size = args.batch_size
test_batch_size = args.batch_size
data = {}

_types = ''

for category in ['train'+_types, 'val'+_types, 'test'+_types]:

    print("# Loading:", category + '.npz')

    # Loading npz
    cat_data = np.load(os.path.join(args.data_cctv, category + '.npz'))

    data['x_' + category] = cat_data['x']     # (?, 12, 207, 2)
    data['y_' + category] = cat_data['y']     # (?, 12, 207, 2)

    print(cat_data['x'].shape)
    print(cat_data['y'].shape)
    '''
    print('x[0]:',cat_data['x'][0])
    print('y[0]:',cat_data['y'][0])
    print('x[-1]',cat_data['x'][-1])
    print('y[-1]',cat_data['y'][-1])
    '''
    print('y[-1]',cat_data['y'][-1])

# 使用train的mean/std來正規化valid/test #
#scaler = StandardScaler(mean=data['x_train'+_types][..., 0].mean(), std=data['x_train'+_types][..., 0].std())
scaler = dataloader_gct['scaler']

# 將欲訓練特徵改成正規化
for category in ['train'+_types, 'val'+_types, 'test'+_types]:
    data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])


data['train_loader'] = DataLoaderM(data['x_train'+_types], data['y_train'+_types], batch_size)
data['val_loader'] = DataLoaderM(data['x_val'+_types], data['y_val'+_types], valid_batch_size)
data['test_loader'] = DataLoaderM(data['x_test'+_types], data['y_test'+_types], test_batch_size)
data['scaler'] = scaler

sensor_ids, sensor_id_to_ind, adj_mx = load_adj(args.adj_data_cctv,args.adjtype)   # adjtype: default='doubletransition'

adj_mx_cctv = [torch.tensor(i).to(device) for i in adj_mx]


dataloader_cctv = data.copy()