import pandas as pd
import numpy as np
from tensorflow.keras.utils import Sequence

def read_data(data_dir, split=True):
    
    data = pd.read_csv(data_dir + 'data.csv', index_col='CaseID')
    
    data_train = pd.read_csv(data_dir + 'train_index.csv', index_col='CaseID')
    data_valid = pd.read_csv(data_dir + 'valid_index.csv', index_col='CaseID')
    data_test = pd.read_csv(data_dir + 'test_index.csv', index_col='CaseID')

    data_train = data[data.index.isin(data_train.index)].copy()
    data_valid = data[data.index.isin(data_valid.index)].copy()
    data_test = data[data.index.isin(data_test.index)].copy()
    
    return data, data_train, data_valid, data_test

def to_integer_vector(vector, output_dim, cut):
    cut_integer_vector = vector[max(0, cut - output_dim):cut]
    cut_integer_vector = [0.0] * (output_dim - len(cut_integer_vector)) + cut_integer_vector
    return cut_integer_vector

def get_x(activity_vector, 
          cat_dic,
          num_dic,
          helpers_dic,
          output_dim, cut):
    
    if len(activity_vector) and isinstance(activity_vector[0], int):
        integer_vector = activity_vector
    else:
        integer_vector = [helpers_dic['trace_helper']['activity_to_integer'][act] for act in activity_vector]
    
    act_cut_integer_vector = integer_vector[max(0, cut - output_dim):cut]
    act_cut_integer_vector = [0.0] * (output_dim - len(act_cut_integer_vector)) + act_cut_integer_vector
    
    return_dic = {}
    for string in ['trace'] + list(cat_dic.keys()) + list(num_dic.keys()):
        if string == 'trace':
            return_dic[f'{string}_seq'] = np.zeros((output_dim, helpers_dic['trace_helper']['vocab_size']))
        else:
            return_dic[f'{string}_seq'] = np.zeros((output_dim, 1))
    
    cat_vector_dic = {}
    for cat_str, cat_vec in cat_dic.items():
        cat_vector_dic[f'{cat_str}_integer_vector'] = to_integer_vector(
            [helpers_dic['cat_helpers'][cat_str][f'{cat_str}_to_integer'][val] for val in cat_vec],
            output_dim, cut)
    
    num_vector_dic = {}
    for num_str, num_vec in num_dic.items():
        num_vector_dic[f'{num_str}_integer_vector'] = to_integer_vector(num_vec, output_dim, cut)        
    
    for p in range(output_dim):
        for v in range(helpers_dic['trace_helper']['vocab_size']):
            return_dic['trace_seq'][p, v] = 1 if act_cut_integer_vector[p] == v else 0
    
    # Process categorical features
    for cat_str, cat_vec in cat_dic.items():
        cat_integer_vector = to_integer_vector(
            [helpers_dic['cat_helpers'][cat_str][f'{cat_str}_to_integer'][val] for val in cat_vec],
            output_dim, cut)
        return_dic[f'{cat_str}_seq'] = np.array(cat_integer_vector).reshape(output_dim, 1)
    
    # Process numerical features
    for num_str, num_vec in num_dic.items():
        num_integer_vector = to_integer_vector(num_vec, output_dim, cut)
        return_dic[f'{num_str}_seq'] = np.array(num_integer_vector).reshape(output_dim, 1)
    
    return return_dic

def get_y(activity_vector, helpers_dic, output_dim, cut):
    if len(activity_vector) and isinstance(activity_vector[0], int):
        integer_vector = activity_vector
    else:
        integer_vector = [helpers_dic['trace_helper']['activity_to_integer'][act] for act in activity_vector]
       
    remtrace_seq = np.zeros((output_dim, helpers_dic['trace_helper']['vocab_size']))
    
    remact_cut_integer_vector = integer_vector[cut:output_dim + cut]
    remact_cut_integer_vector = remact_cut_integer_vector + [0.0] * (output_dim - len(remact_cut_integer_vector))
    
    for p in range(output_dim):
        for v in range(helpers_dic['trace_helper']['vocab_size']):
            remtrace_seq[p, v] = 1 if remact_cut_integer_vector[p] == v else 0
    
    return remtrace_seq

def get_batch(data_frame, selected_ids, output_dim, feat_dic, helpers_dic, selected_cuts_or_strategy='random', return_indexes=False):
    
    cat_feat = feat_dic['cat_feat']
    num_feat = feat_dic['num_feat']
    
    X_dic = {}
    for col in ['trace'] + cat_feat + num_feat:
        X_dic[f'X{col}'] = []  
    
    YTrace = []
    idxs = []
    
    for i, idx in enumerate(selected_ids):
        activity_vector = data_frame.loc[idx, 'trace'].split(', ')     
        
        cat_dic = {col: data_frame.loc[idx, col].split(', ') if isinstance(data_frame.loc[idx, col], str) else data_frame.loc[idx, col] for col in cat_feat}
        
        num_dic = {}
        for col in num_feat:
            value = data_frame.loc[idx, col]
            if isinstance(value, list):
                num_dic[col] = [float(n) for n in value]
            elif isinstance(value, str):
                # Remove square brackets and split the string
                value = value.strip('[]')
                num_dic[col] = [float(n) for n in value.split(', ')]
            elif isinstance(value, (int, float)):
                num_dic[col] = [float(value)]
            else:
                # Handle or raise exception
                num_dic[col] = []
        
        if selected_cuts_or_strategy == 'random':
            cut_list = [np.random.randint(1, len(activity_vector) + 1)] 
        elif selected_cuts_or_strategy == 'all':
            cut_list = range(1, len(activity_vector) + 1)
        else:
            cut_list = selected_cuts_or_strategy[i]
        
        for cut in cut_list:
            return_dic = get_x(activity_vector, 
                               cat_dic, 
                               num_dic, 
                               helpers_dic,
                               output_dim, cut)
    
            for key in list(X_dic.keys()):
                X_dic[key].append(return_dic[f'{key.replace("X", "")}_seq'])
            
            Trace_out = get_y(activity_vector, helpers_dic, output_dim, cut)
            YTrace.append(Trace_out)
            
            idxs.append(idx)
    
    X = {}
    X['trace_input'] = np.array(X_dic['Xtrace'])  # Shape: (batch_size, output_dim, vocab_size)
    
    # Concatenate categorical features along the last axis
    cat_feature_keys = [key for key in X_dic.keys() if key.startswith('Xcat')]
    cat_features = [np.array(X_dic[key]) for key in cat_feature_keys]
    X['cat_input'] = np.concatenate(cat_features, axis=-1)  # Shape: (batch_size, output_dim, num_cat_features)
    
    # Concatenate numerical features along the last axis
    num_feature_keys = [key for key in X_dic.keys() if key.startswith('Xnum')]
    num_features = [np.array(X_dic[key]) for key in num_feature_keys]
    X['num_input'] = np.concatenate(num_features, axis=-1)  # Shape: (batch_size, output_dim, num_num_features)
    
    Y = {'trace_out': np.array(YTrace)}
    IDs = {'Bag_ids': np.array(idxs)}
          
    if return_indexes:
        return X, Y, IDs
    else:
        return X, Y


class BagDataGenerator(Sequence):
    def __init__(self, data_frame, output_dim, feat_dic, helpers_dic, batch_size=128, shuffle=True, override_indexes=None, **batch_kwargs): 
        self.data_frame = data_frame
        self.indexes = np.array(self.data_frame.index) if override_indexes is None else override_indexes
        self.output_dim = output_dim
        self.feat_dic = feat_dic
        self.helpers_dic = helpers_dic
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.batch_kwargs = batch_kwargs

    def __len__(self): 
        return int(np.ceil(self.indexes.shape[0] / self.batch_size))

    def __getitem__(self, index): 
        selected_ids = self.indexes[index * self.batch_size : (index + 1) * self.batch_size] 
        return self.__data_generation(selected_ids)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, selected_ids): 
        return get_batch(self.data_frame, selected_ids, self.output_dim, self.feat_dic, self.helpers_dic, **self.batch_kwargs)
