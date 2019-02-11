import tensorflow as tf 
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import warnings
warnings.filterwarnings('ignore') 

data_file = '.\CTR\trains.csv'
data = pd.read_csv(data_file)

#此份数据为https://github.com/htbeker/LR_GBDT项目缩减版
bin_features = [ ]
cat_features = [ ]
other_features = [ ]
for i in data.columns.tolist():
    if 'bin' in i:
        bin_features.append(i)
    elif 'cat' in i:
        cat_features.append(i)
    elif 'id' != i and 'target' != i:
        other_features.append(i)        
print(bin_features)
print(cat_features)
print(other_features)
_CSV_COLUMNS = ['target', 'ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03',
       'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_ind_06_bin', 'ps_ind_07_bin',
       'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01',
       'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat',
       'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14']
    
def input_fn(data_file, num_epochs,shuffle):
    df_data = pd.read_csv(
        tf.gfile.Open(data_file),
        names=_CSV_COLUMNS,
        skipinitialspace=True,
        engine="python",
        skiprows=1)
    labels = df_data["target"]
    df_data = df_data.drop("target",axis=1)

    return tf.estimator.inputs.pandas_input_fn(
        x=df_data,
        y=labels,
        batch_size=100,
        num_epochs=num_epochs,
        shuffle = shuffle,
        num_threads=5)
#使用tf.feature_column分别对离散和连续特征进行处理
#连续特征在wide和deep部分都会用到
ps_ind_01 = tf.feature_column.numeric_column('ps_ind_01')
ps_ind_03 = tf.feature_column.numeric_column('ps_ind_03')
ps_reg_01 = tf.feature_column.numeric_column('ps_reg_01')
ps_calc_10 = tf.feature_column.numeric_column('ps_calc_10')
ps_calc_11 = tf.feature_column.numeric_column('ps_calc_11')
ps_calc_12 = tf.feature_column.numeric_column('ps_calc_12')
ps_calc_13 = tf.feature_column.numeric_column('ps_calc_13')
ps_calc_14 = tf.feature_column.numeric_column('ps_calc_14')
#离散特征
ps_ind_06_bin = tf.feature_column.categorical_column_with_identity(key = 'ps_ind_06_bin',num_buckets=2)
ps_ind_07_bin = tf.feature_column.categorical_column_with_identity(key = 'ps_ind_07_bin',num_buckets=2)
ps_ind_16_bin = tf.feature_column.categorical_column_with_identity(key = 'ps_ind_16_bin',num_buckets=2)
ps_ind_17_bin = tf.feature_column.categorical_column_with_identity(key = 'ps_ind_17_bin',num_buckets=2)
ps_ind_18_bin = tf.feature_column.categorical_column_with_identity(key = 'ps_ind_18_bin',num_buckets=2)
ps_ind_02_cat = tf.feature_column.categorical_column_with_vocabulary_list(
    key = 'ps_ind_02_cat',vocabulary_list = [ 2,  1,  4,  3, -1])
ps_ind_04_cat = tf.feature_column.categorical_column_with_vocabulary_list(
    key = 'ps_ind_04_cat',vocabulary_list = [1, 0, -1])
ps_ind_05_cat = tf.feature_column.categorical_column_with_vocabulary_list(
    key = 'ps_ind_05_cat',vocabulary_list = [ 0,1,4,3,6 ,5,-1,2])
ps_car_03_cat = tf.feature_column.categorical_column_with_vocabulary_list(
    key = 'ps_car_03_cat',vocabulary_list = [-1,0,1])
ps_car_04_cat = tf.feature_column.categorical_column_with_vocabulary_list(
    key = 'ps_car_04_cat',vocabulary_list = [0, 1, 8, 9, 2,6, 3, 7, 4, 5])
ps_car_05_cat = tf.feature_column.categorical_column_with_vocabulary_list(
    key = 'ps_car_05_cat',vocabulary_list = [ 1, -1,0])
ps_car_06_cat = tf.feature_column.categorical_column_with_vocabulary_list(
    key = 'ps_car_06_cat',vocabulary_list = [4,11,14,13,6,15,3,0,1,10,12,9,17,7,8,5,2,16])
#交叉特征
crossed_columns = [
    tf.feature_column.crossed_column(
        ['ps_ind_02_cat', 'ps_car_03_cat'], hash_bucket_size=100),
    tf.feature_column.crossed_column(
        ['ps_ind_02_cat', 'ps_car_04_cat', 'ps_car_05_cat'], hash_bucket_size=100
    ),
    tf.feature_column.crossed_column(
    ['ps_car_03_cat', 'ps_car_06_cat'], hash_bucket_size=100)
]
#在学习过程中线性模型接受所有类型特征列，深度神经网络分类器DNNClassifier仅接收密集特征列dense column,
#其他类型特征列必须用指示列indicatorColumn或嵌入列embedingColumn进行包裹
raw_input_col = [ ps_ind_06_bin ,
      ps_ind_07_bin ,
      ps_ind_16_bin ,
      ps_ind_17_bin ,
      ps_ind_18_bin ,
      ps_ind_02_cat ,
      ps_ind_04_cat ,
      ps_ind_05_cat ,
      ps_car_03_cat ,
      ps_car_04_cat ,
      ps_car_05_cat ,
      ps_car_06_cat
    ]

deep_columns = [
    ps_ind_01 ,
    ps_ind_03 ,
    ps_reg_01 ,
    ps_calc_10 ,
    ps_calc_11 ,
    ps_calc_12 ,
    ps_calc_13 ,
    ps_calc_14 ,
    tf.feature_column.indicator_column(ps_ind_06_bin),
    tf.feature_column.indicator_column(ps_ind_07_bin),
    tf.feature_column.indicator_column(ps_ind_16_bin),
    tf.feature_column.indicator_column(ps_ind_17_bin),
    tf.feature_column.indicator_column(ps_ind_18_bin),
    tf.feature_column.indicator_column(ps_ind_02_cat),
    tf.feature_column.indicator_column(ps_ind_04_cat),
    tf.feature_column.indicator_column(ps_ind_05_cat),
    tf.feature_column.indicator_column(ps_car_03_cat),
    tf.feature_column.indicator_column(ps_car_04_cat),
    tf.feature_column.indicator_column(ps_car_05_cat),
    tf.feature_column.embedding_column(ps_car_06_cat,dimension= 5)#此处是为了表示embedding_column的作用，在维度较大时使用
]
#Wide&Deep模型
model_dir = './DeepLearning/wide_deep'
model = tf.estimator.DNNLinearCombinedClassifier(model_dir = model_dir,
                                                linear_feature_columns = raw_input_col+crossed_columns,
                                                dnn_feature_columns = deep_columns,
                                                dnn_hidden_units = [100,50])

train_epochs = 6
train_file = '.\CTR\trains.csv'
test_file = '.\CTR\tests.csv'
model.train(input_fn= input_fn(train_file, train_epochs,True))
results = model.evaluate(input_fn= input_fn(test_file, epochs_per_eval,True))
for key in sorted(results):
    print("%s: %s" % (key, results[key]))
