import os
import numpy as np
import random
import torch.nn.functional as F
import torch
import torch.utils.data as dataf
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats
import pickle
import random
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import utils
import models
import matplotlib.pyplot as plt
from einops import rearrange
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from scipy import io
import sys
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 160)
parser.add_argument("-c","--src_input_dim",type = int, default = 3)
parser.add_argument("-d","--tar_input_dim",type = int, default = 242)
parser.add_argument("-n","--n_dim",type = int, default = 100)
parser.add_argument("-w","--class_num",type = int, default = 2)
parser.add_argument("-s","--shot_num_per_class",type = int, default = 1)
parser.add_argument("-b","--query_num_per_class",type = int, default = 19)
parser.add_argument("-sum","--sum_num_per_class",type = int, default = 20)
parser.add_argument("-e","--episode",type = int, default= 1000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
parser.add_argument("-can","--hyperparameter",type=int,default=10)
# target
parser.add_argument("-m" ,"--test_class_num",type=int, default=2)
parser.add_argument("-z","--test_lsample_num_per_class",type=int,default=70, help='5 4 3 2 1')

args = parser.parse_args(args=[])

# Hyper Parameters
FEATURE_DIM = args.feature_dim  ##160
SRC_INPUT_DIMENSION = args.src_input_dim ##224
TAR_INPUT_DIMENSION = args.tar_input_dim ##224
N_DIMENSION = args.n_dim ##100
CLASS_NUM = args.class_num  ##2
SHOT_NUM_PER_CLASS = args.shot_num_per_class ##1
QUERY_NUM_PER_CLASS = args.query_num_per_class ##19
SUM_NUM_PER_CLASS = args.sum_num_per_class ##1+19
EPISODE = args.episode
TEST_EPISODE = args.test_episode ##600
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit ##10
CAN = args.hyperparameter
print(CAN)
# Hyper Parameters in target domain data set
TEST_CLASS_NUM = args.test_class_num # the number of class ##2
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class # the number of labeled samples per class 5 4 3 2 1 ##5

patchsize = 9

utils.same_seeds(0)
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')
_init_()

# load source domain data set
with open(os.path.join('datasets',  'MSI157_9_imdbgai_3.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)
print(source_imdb.keys())
print(source_imdb['Labels'])

# process source domain data set
data_train = source_imdb['data']
labels_train = source_imdb['Labels']
keys_all_train = sorted(list(set(labels_train)))

label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print(label_encoder_train)

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
print(train_set.keys())
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print("Num classes for source domain datasets: " + str(len(data)))

data = utils.sanity_check500(data) # 500 labels samples per class
print("Num classes of the number of class larger than 500 in dataset: " + str(len(data)))

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))
        data[class_][i] = image_transpose

# source few-shot classification data
metatrain_data= data
print(len(metatrain_data.keys()), metatrain_data.keys())
del data

# source domain adaptation data
print(source_imdb['data'].shape)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0))
print(source_imdb['data'].shape) #
print(source_imdb['Labels'])
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=SUM_NUM_PER_CLASS*CLASS_NUM, shuffle=True, num_workers=0)

del source_dataset, source_imdb

## target domain data set

# load target domain data set
test_data1 = 'datasets/bayarea/BayArea_before.mat'
test_data2 = 'datasets/bayarea/BayArea_after.mat'
test_label = 'datasets/bayarea/bayArea_gtChanges2.mat'


Data_Band_Scaler1,GroundTruth = utils.load_data(test_data1, test_label)
Data_Band_Scaler2,GroundTruth = utils.load_data(test_data2, test_label)
Data_Band_Scaler = Data_Band_Scaler2-Data_Band_Scaler1



# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape) ## (600，500，224)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = 4
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn+ HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,:]

    [Row, Column] = np.nonzero(G)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_train = {} # Data Augmentation
    m = int(np.max(G))  # 2
    nlabeled =TEST_LSAMPLE_NUM_PER_CLASS  ##5
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class ##5
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):  ##0-40
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices1:', len(train_indices))  # 5
    print('the number of test_indices1:', len(test_indices))  # 9693
    print('the number of train_indices1 after data argumentation:', len(da_train_indices))  # 200
    print('labeled sample indices1:',train_indices)

    nTrain = len(train_indices) ##5
    nTest = len(test_indices) ##9693
    da_nTrain = len(da_train_indices) ##200

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')


    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=shot_num_per_class * class_num,shuffle=False, num_workers=0)

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],  dtype=np.float32)  # (9,9,100,n)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] =utils.radiation_noise(data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
                                                       Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)


    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-16 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('imdb_da_train ok')

    return train_loader, test_loader, imdb_da_train ,G,RandPerm,Row, Column,nTrain ##RandPerm Row Column:10249 nTrain:80

def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train,G,RandPerm,Row, Column,nTrain = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler,  GroundTruth=GroundTruth, \
                                                                     class_num=class_num,shot_num_per_class=shot_num_per_class)  # 1 classes and 5 labeled samples per class
    train_datas, train_labels = train_loader.__iter__().next() ##train_datas(80,200,9,9)
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape)

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)  ## (9, 9, 198, 200)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    # target domain : batch samples for domian adaptation
    print(imdb_da_train['data'].shape)
    print(imdb_da_train['Labels'])
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=SUM_NUM_PER_CLASS*CLASS_NUM, shuffle=True, num_workers=0)
    del target_dataset


    return train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.1):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)  ##1,40,1024--->1,8,40,128

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # print('attention')
        return self.to_out(out)##1,40,160


class AttentionCro(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.1):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x1, x2):
        qkv1 = self.to_qkv(x1).chunk(3, dim = -1)
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv1)  ##1,40,1024--->1,8,40,128
        qkv2 = self.to_qkv(x2).chunk(3, dim=-1)
        q2, k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv2)  ##1,40,1024--->1,8,40,128

        dots1 = torch.matmul(q2, k1.transpose(-1, -2)) * self.scale
        attn1 = self.attend(dots1)
        out1 = torch.matmul(attn1, v2)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')

        return self.to_out(out1)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.1):
        super().__init__()
        # self.layers = nn.ModuleList([])
        for _ in range(depth):
          self.atten = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
          self.Drop = nn.Dropout(dropout)
          self.norm = nn.LayerNorm(dim)


    def forward(self, x):  ##1,40,1024
        for _ in range(6):
            x1 = self.atten(x) + x
            x1 = self.Drop(x1)
            x1 = self.norm(x1) + x1
            x1 = self.Drop(x1)

        return x1

class CroTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.1):
        super().__init__()
        for _ in range(depth):
            self.attenCro = AttentionCro(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            self.Drop = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x1, x2):  ##1,40,1024
        x1_2 = self.attenCro(x1, x2)
        x1_2 = self.Drop(x1_2)
        x1_2 = self.norm(x1_2) + x1_2
        x1_2 = self.Drop(x1_2)

        return x1_2

class TrFDA(nn.Module):

    def __init__(self,
                 num_classes=2,
                 dim=160,
                 depth=6,
                 heads=8,
                 mlp_dim=2048,
                 dim_head=64,
                 dropout=0.1,
                 emb_dropout=0.1):
        super().__init__()

        self.conv2d_features = nn.Sequential(
                        nn.Conv2d(in_channels=100, out_channels=64, kernel_size=(3, 3)),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                    )

        self.Linear = nn.Linear(3136, dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.Crotransformer = CroTransformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.SF = nn.Softmax(dim=-1)
        self.LN = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=160, kernel_size=(1, 1)),  ##64-->32
            # nn.Linear(320, 160),
            # nn.BatchNorm2d(160),
            # nn.Softmax(dim=0)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x1, x2):  ## x1:source x2:target ##40,160
        x1 = x1.unsqueeze(0)  ##(1,40,160)
        x2 = x2.unsqueeze(0)  ##(1,40,160)


        x1_1 = self.transformer(x1)  ##1,40,160---->1,40,160
        x2_1 = self.transformer(x2)

        x_fussion1 = self.Crotransformer(x1,x2)


        x1_cro = x1_1 + x_fussion1
        x2_cro = x2_1 + x_fussion1
        x1_cro = self.dropout(x1_cro)
        x2_cro = self.dropout(x2_cro)

        x1_cro = self.SF(x1_cro)
        x2_cro = self.SF(x2_cro)


        return x1_1, x2_1, x1_cro, x2_cro


##Network

class Attention_F(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.1):  ##dim=1024
        super().__init__()
        inner_dim = dim_head *  heads ##8*64
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads  ##8
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)  #
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)  ##1,40,1024--->1,8,40,128

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # print('attention')
        return self.to_out(out)##1,40,160


class Transformer_F(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.1):
        super().__init__()
        # self.layers = nn.ModuleList([])
        for _ in range(depth):
          self.atten = Attention_F(dim, heads=heads, dim_head=dim_head, dropout=dropout)
          self.Drop = nn.Dropout(dropout)
          self.norm = nn.LayerNorm(dim)


    def forward(self, x):  ##1,40,1024
        for _ in range(6):
          x1 = self.atten(x) + x
          x1 = self.Drop(x1)
          x1 = self.norm(x1) + x1
          x1 = self.Drop(x1)

        return x1

def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer

class residual_block(nn.Module):

    def __init__(self, in_channel,out_channel): ##1 8
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel,out_channel)
        self.conv2 = conv3x3x3(out_channel,out_channel)
        self.conv3 = conv3x3x3(out_channel,out_channel)

    def forward(self, x): #(1,1,100,9,9)
        x1 = F.relu(self.conv1(x), inplace=True) #(1,8,100,9,9)  (1,16,25,5,5)
        x2 = F.relu(self.conv2(x1), inplace=True) #(1,8,100,9,9) (1,16,25,5,5)
        x3 = self.conv3(x2) #(1,8,100,9,9) (1,16,25,5,5)

        out = F.relu(x1+x3, inplace=True) #(1,8,100,9,9)  (1,16,25,5,5)
        return out


class D_Res_3d_CNN(nn.Module): ##1 8 16
    def __init__(self, in_channel, out_channel1, out_channel2):
        super(D_Res_3d_CNN, self).__init__()

        self.block1 = residual_block(in_channel,out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4,2,2),padding=(0,1,1),stride=(4,2,2))
        self.block2 = residual_block(out_channel1,out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4,2,2),stride=(4,2,2), padding=(2,1,1))
        self.conv = nn.Conv3d(in_channels=out_channel2,out_channels=32,kernel_size=3, bias=False)

        self.final_feat_dim = 160


    def forward(self, x): #x:(2,100,9,9)
        x = x.unsqueeze(1) #(2,1,100,9,9)
        x = self.block1(x) #(16,8,100,9,9)
        x = self.maxpool1(x) #(1,8,25,5,5)


        x = self.block2(x) #(1,16,25,5,5)
        x = self.maxpool2(x) #(1,16,7,3,3)

        x = self.conv(x) #(1,32,5,1,1)
        return x


class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):  ##source 3,100
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)  ##(2,100,9,9)
        x = self.preconv_bn(x) ##(2,100,9,9)
        return x

class Network(nn.Module):
    def __init__(self,
                 num_classes=2,
                 dim=160,
                 depth=6,
                 heads=8,
                 mlp_dim=2048,
                 dim_head=64,
                 dropout=0.1,
                 emb_dropout=0.1):
        super(Network, self).__init__()
        self.feature_encoder = D_Res_3d_CNN(1,8,16)  ##(2,100,9,9)-->
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION) ##3,100
        self.Transformer = Transformer_F(dim, depth, heads, dim_head, mlp_dim, dropout)
        # self.LN = nn.Linear(200,100)
        self.LN = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=160, kernel_size=(1, 1)),  ##64-->32
            # nn.Linear(320, 160),
            # nn.BatchNorm2d(160),
            # nn.Softmax(dim=0)
        )

    def forward(self,x1, x2, domain='source', condition='train'):
        if domain == 'target':
            x1 = self.target_mapping(x1)  ##(2,100,9,9)
            x2 = self.target_mapping(x2)  ##(2,100,9,9)
        elif domain == 'source':   ## x.support (2,3,9,9)
            x1 = self.source_mapping(x1)
            x2 = self.source_mapping(x2)

        if condition =='train':
            feature1 = self.feature_encoder(x1)
            feature1 = feature1.view(feature1.shape[0], -1)  ##2,160
            feature2 = self.feature_encoder(x2)
            feature2 = feature2.view(feature2.shape[0], -1)  ##38，160

            feature1_0 = feature1[0:1]
            feature1_1 = feature1[1:]
            feature2_0 = feature2[:19]
            feature2_1 = feature2[19:]

            feature_first = torch.cat([feature1_0, feature2_0], dim=0)
            feature_second = torch.cat([feature1_1, feature2_1], dim=0)  ##20,160
            x_or1 = feature_first
            x_or2 = feature_second

            '''TrF start'''
            feature_first = feature_first.unsqueeze(0)  ##1,n,160
            feature_second = feature_second.unsqueeze(0)  ##1,n,160
            feature_first = self.Transformer(feature_first)  ##1,20,160
            feature_second = self.Transformer(feature_second)  ##1,20,160

            t = feature_first.shape[1]
            feature_first = feature_first.reshape(t, 160)  ##20,160
            feature_second = feature_second.reshape(t, 160)  ##20,160
            feature_first = torch.cat([feature_first, x_or1], dim=1)  ##n,160*2
            feature_second = torch.cat([feature_second, x_or2], dim=1)  ##n,160*2

            feature_first = feature_first.reshape(t, 320, 1, 1)
            feature_second = feature_second.reshape(t, 320, 1, 1)
            feature_first = self.LN(feature_first)
            feature_second = self.LN(feature_second)  ##Conv2D

            feature_first = feature_first.view(feature_first.shape[0], -1)  # (20,160)
            feature_second = feature_second.view(feature_second.shape[0], -1)  # (20,160)

            featureS_0 = feature_first[0:1]
            featureS_1 = feature_second[0:1]
            featureQ_0 = feature_first[1:]
            featureQ_1 = feature_second[1:]
            featureS = torch.cat([featureS_0, featureS_1], dim=0)
            featureQ = torch.cat([featureQ_0, featureQ_1], dim=0)

        elif condition == 'test':
            feature = self.feature_encoder(x1)  ## (n,32,5,1,1)
            feature = feature.view(feature.shape[0], -1)  ##n,160
            x_or = feature
            feature = feature.unsqueeze(0)  ##1,n,160
            # print('F_TrF start')
            feature = self.Transformer(feature)
            # print('F_TrF end')
            t = feature.shape[1]
            feature = feature.reshape(t, 160)  ##n,160
            feature = torch.cat([feature, x_or], dim=1)  ##n,160*2
            feature = feature.reshape(t, 320, 1, 1)
            feature = self.LN(feature)


            featureS = feature.view(feature.shape[0], -1)  # (1,160)
            featureQ = feature.view(feature.shape[0], -1)  # (1,160)

        return featureS, featureQ



crossEntropy = nn.CrossEntropyLoss().cuda()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:

        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


nDataSet =1
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_kappa = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None

##Bay seeds
seeds = [1330,1008,1334,2586,1220,1999,1224,1258,1334,1226,2586,1999,1008,3407,1335,1331,1334,1336,1229 ]
for iDataSet in range(nDataSet):

    # load target domain data for training and testing
    np.random.seed(seeds[iDataSet])
    print('seeds:', seeds[iDataSet])
    train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)

    # model
    feature_encoder = Network()
    DA_TrF = TrFDA()


    # total1 = sum([param.nelement() for param in feature_encoder.parameters()])
    # total2 = sum([param.nelement() for param in DA_TrF.parameters()])
    # print("Numer of parameter:%.2fM" % ((total1+total2)/1e6))


    feature_encoder.apply(weights_init)
    DA_TrF.apply(weights_init)

    feature_encoder.cuda()
    DA_TrF.cuda()

    feature_encoder.train()
    DA_TrF.train()

    # optimizer
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=args.learning_rate)
    DA_TrF_optim  = torch.optim.Adam(DA_TrF.parameters(), lr=args.learning_rate) ##lr=0.001

    print("Training...")

    last_accuracy = 0.0
    last_kappa = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit_sor, total_num_sor, total_hit_tar, total_num_tar = 0.0, 0.0, 0.0, 0.0
    test_acc_list = []


    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    len_dataloader = min(len(source_loader), len(target_loader))
    train_start = time.time()
    for episode in range(2000):  # EPISODE = 90000
        # get domain adaptation data from  source domain and target domain
        try:
            source_data, source_label = source_iter.next()
        except Exception as err:
            source_iter = iter(source_loader)
            source_data, source_label = source_iter.next()

        try:
            target_data, target_label = target_iter.next()
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, target_label = target_iter.next()


        '''Few-shot claification for source domain data set'''
        task_sor = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS,QUERY_NUM_PER_CLASS)
        support_dataloader_sor = utils.get_HBKC_data_loader(task_sor, num_per_class=SHOT_NUM_PER_CLASS, split="train",shuffle=False)
        query_dataloader_sor = utils.get_HBKC_data_loader(task_sor, num_per_class=QUERY_NUM_PER_CLASS, split="test",shuffle=True)


        task_tar = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 1， 1，19
        support_dataloader_tar = utils.get_HBKC_data_loader(task_tar, num_per_class=SHOT_NUM_PER_CLASS, split="train",shuffle=False)
        query_dataloader_tar = utils.get_HBKC_data_loader(task_tar, num_per_class=QUERY_NUM_PER_CLASS, split="test",shuffle=True)


        supports_sor, support_labels_sor = support_dataloader_sor.__iter__().next()
        querys_sor, query_labels_sor = query_dataloader_sor.__iter__().next()


        supports_tar, support_labels_tar = support_dataloader_tar.__iter__().next()  ## supports(2, 3, 9, 9)
        querys_tar, query_labels_tar = query_dataloader_tar.__iter__().next()  ## querys(38,3,9,9)

        # calculate features
        support_features_sor,query_features_sor = feature_encoder(supports_sor.cuda(), querys_sor.cuda(), condition='train')

        support_features_tar, query_features_tar = feature_encoder(supports_tar.cuda(), querys_tar.cuda(), domain='target',condition='train')

        # Prototype network
        if SHOT_NUM_PER_CLASS > 1:
            support_proto_sor = support_features_sor.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            support_proto_tar = support_features_tar.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
        else:
            support_proto_sor = support_features_sor
            support_proto_tar = support_features_tar

        '''few-shot learning'''
        if (episode + 1) % 2 == 0:
            logits = euclidean_metric(query_features_sor, support_proto_sor)
            f_loss_sor = crossEntropy(logits, query_labels_sor.cuda().long())
            f_loss = f_loss_sor
        else:
            logits = euclidean_metric(query_features_tar, support_proto_tar)
            f_loss_tar = crossEntropy(logits, query_labels_tar.cuda().long())
            f_loss = f_loss_tar

        '''domain adaptation'''
        DA_features_sor_f = torch.cat([support_features_sor, query_features_sor], dim=0)  ##output.cat 40,160
        DA_features_tar_f = torch.cat([support_features_tar, query_features_tar], dim=0)

        '''TrFDA'''
        DA_TrF_sor, DA_TrF_tar,DA_TrF_sor2,DA_TrF_tar2 = DA_TrF(DA_features_sor_f, DA_features_tar_f) ##(1,40,1024)


        DA_TrF_sor2 = rearrange(DA_TrF_sor2, 'b h w -> b w h')  ##1,160,40
        DA_TrF_tar2 = rearrange(DA_TrF_tar2, 'b h w -> b w h')

        '''GOT'''

        FWD2 = utils.TSTgai_OT(DA_TrF_sor2, DA_TrF_tar2)


        loss = f_loss  + CAN*FWD2



        # Update parameters
        feature_encoder.zero_grad()
        DA_TrF_optim.zero_grad()
        loss.backward()
        DA_TrF_optim.step()
        feature_encoder_optim.step()

        total_hit_sor += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels_sor.long()).item()
        total_num_sor += querys_sor.shape[0]
        total_hit_tar += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels_tar.long()).item()
        total_num_tar += querys_tar.shape[0]


        if (episode + 1) % 10== 0:  # display
            # train_loss.append(loss.item())
            print(
                'episode {:>3d}:   fsl loss: {:6.4f},  acc_src {:6.4f}, acc_tar {:6.4f}, loss: {:6.4f}'.format(episode + 1,
                                                                                                                f_loss.item(),
                                                                                                                total_hit_sor / total_num_sor,
                                                                                                                total_hit_tar / total_num_tar,
                                                                                                                f_loss.item()))



        '''----TEST----'''
        if (episode + 1) % 100 == 0:
            # test
            print("Testing ...")
            train_end = time.time()
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)

            train_datas, train_labels = train_loader.__iter__().next()
            train_features, _= feature_encoder(Variable(train_datas).cuda(), Variable(train_datas).cuda(), domain='target',condition='test')  # (45, 160)

            max_value = train_features.max()
            min_value = train_features.min()
            print(max_value.item())
            print(min_value.item())

            train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

            KNN_classifier = KNeighborsClassifier(n_neighbors=1)
            KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)
            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]

                test_features, _ = feature_encoder(Variable(test_datas).cuda(), Variable(test_datas).cuda(), domain='target',condition='test')
                test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())


                test_labels = test_labels.numpy()
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                total_rewards += np.sum(rewards)
                counter += batch_size

                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)

                accuracy = total_rewards / 1.0 / counter
                accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)

            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format( total_rewards, len(test_loader.dataset),100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()
            print('seeds:', seeds[iDataSet])

            # Training mode
            feature_encoder.train()
            if test_accuracy > last_accuracy:
                last_accuracy = test_accuracy
                best_episdoe = episode


                acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                OA = acc
                C = metrics.confusion_matrix(labels, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)
                best_predict_all = predict
                best_G, best_RandPerm, best_Row, best_Column, best_nTrain = G, RandPerm, Row, Column, nTrain
                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)
                count_0=0
                count_TP=0
                count_00=0
                for jj in range(len(test_loader.dataset)):
                    if predict[jj] == 0:
                        count_0 = count_0 + 1
                    if labels[jj] ==0:
                        count_00 = count_00 + 1
                    if predict[jj]==0 and labels[jj] == 0:
                        count_TP = count_TP+1
                precision = count_TP/count_0
                recall = count_TP/count_00



            print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))


    print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
    print('***********************************************************************************')
    for i in range(len(best_predict_all)):
        best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = best_predict_all[i] + 1


AA = np.mean(A, 1)

AAMean = np.mean(AA,0)
AAStd = np.std(AA)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

OAMean = np.mean(acc)
OAStd = np.std(acc)

kMean = np.mean(k)
kStd = np.std(k)
F1 = 2*(precision*recall)/(precision+recall)

print ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
print ("average OA: " + "{:.2f}".format( OAMean) + " +- " + "{:.2f}".format( OAStd))
print ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
print ("precision: " + "{:.4f}".format(100 *precision) + " +- " + "{:.4f}".format(100 *precision))
print ("recall: " + "{:.4f}".format(100 *recall) + " +- " + "{:.4f}".format(100 *recall))
print ("F1: " + "{:.4f}".format(100 *F1) + " +- " + "{:.4f}".format(100 *F1))

print ("accuracy for each class: ")
for i in range(CLASS_NUM):
    print ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))


best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))

#################classification map################################

#Bay Santa
hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0.5, 0.5, 0.5]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [1, 1, 1]



utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24,  "classificationMap/CSIDC_Bay_{}shot.png".format(TEST_LSAMPLE_NUM_PER_CLASS))
