"""### Parameter"""

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda',help='')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')


parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=str_to_bool, default=True,help='whether to construct adaptive adjacency matrix')
parser.add_argument('--cl', type=str_to_bool, default=True,help='whether to do curriculum learning')

parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')

parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')

parser.add_argument('--conv_channels',type=int,default=32,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=32,help='residual channels')


parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=12,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=12,help='output sequence length')


parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--clip',type=int,default=5,help='clip')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='adj alpha')

##### GWNET's diff parameter #####
parser.add_argument('--model_type',type=str,default='GWNET',help='model type')
parser.add_argument('--skip_channels',type=int,default=256,help='skip channels')
parser.add_argument('--end_channels',type=int,default=512,help='end channels')
parser.add_argument('--layers',type=int,default=8,help='number of layers')
parser.add_argument('--kernel_set',default=[2], type=int, nargs='+')

parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=101,help='random seed')
parser.add_argument('--save',type=str,default='./save/',help='save path')

parser.add_argument('--log_print', type=str_to_bool, default=False ,help='whether to load static feature')

parser.add_argument('--learning_rate',type=float,default=0.0005,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')

parser.add_argument('--step_size1',type=int,default=400,help='step_size')
parser.add_argument('--step_size2',type=int,default=100,help='step_size')


#------------------#

### GCT ###
target = 'hsin_49_GCT_0600_1900_rename'
parser.add_argument('--data_gct',type=str,default='../Data/'+target ,help='data path')
parser.add_argument('--adj_data_gct',type=str,default='../Data/'+target+'/adj_mat_49_corrected.pkl',help='adj data path')
parser.add_argument('--num_nodes_gct',type=int,default=49,help='number of nodes/variables')
parser.add_argument('--expid_gct',type=int,default=202312151449,help='experiment id')
parser.add_argument('--subgraph_size_gct',type=int,default=20,help='k')

### CCTV ###
target = 'hsin_9_CCTV_0600_1900_rename'
parser.add_argument('--data_cctv',type=str,default='../Data/'+target ,help='data path')
parser.add_argument('--adj_data_cctv',type=str,default='../Data/'+target+'/adj_mat_9.pkl',help='adj data path')
parser.add_argument('--num_nodes_cctv',type=int,default=9,help='number of nodes/variables')
parser.add_argument('--expid_cctv',type=int,default=202301041709,help='experiment id')
parser.add_argument('--subgraph_size_cctv',type=int,default=4,help='k')

# cam ID: 5	4	1	2	3	6	7	8	9 -> corre road ID: 45 27 5 7 16 42 37 30 22 -> corre idx: 7 21 19 17 3 15 1 6 9
parser.add_argument('--all_idx',type=int,default=[0, 1, 2, 3, 4, 5, 6, 7, 8],help='idxs')
parser.add_argument('--corre_idx',type=int,default=[7, 21, 19, 17, 3, 15, 1, 6, 9],help='idxs')
parser.add_argument('--exclue_idx',type=int,default=[4],help='idxs')

### Fusion ###
parser.add_argument('--expid_fusion',type=int,default=202401251640,help='experiment id')
parser.add_argument('--runs',type=int,default=10,help='number of runs')
parser.add_argument('--epochs',type=int,default=180,help='')

args=parser.parse_args(args=[])
torch.set_num_threads(3)

args=parser.parse_args(args=[])
print('# args', args)

device = torch.device(args.device)

writer = SummaryWriter()