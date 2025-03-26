import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import numpy as np



def remove_cols(panda,columns,inplace):
    return panda.drop(columns=columns,inplace=inplace)

def choose_cols(panda,columns):
    return panda[columns].copy()

def choose_intCols(panda):
    return panda.select_dtypes(include=["int64"])

def choose_floatCols(panda):
    return panda.select_dtypes(include=["float64"])

def choose_binCols(panda):
    int_cols=choose_intCols(panda)

    binary_columns=[col for col in int_cols.columns if int_cols[col].nunique() == 2]

    return int_cols[binary_columns]

def choose_nonBinCols(panda):
    int_cols=choose_intCols(panda)

    #non_binary_int_columns = [col for col in int_cols.columns if int_cols[col].nunique() != 2]
    #return int_cols[non_binary_int_columns]
    non_binary_int_cols = [col for col in int_cols.columns if int_cols[col].nunique() != 2]
    df_non_binary = int_cols[non_binary_int_cols]
    
    # Split based on the number of unique values
    categorical_cols = [col for col in df_non_binary.columns if df_non_binary[col].nunique() <= 10]
    continuous_cols = [col for col in df_non_binary.columns if df_non_binary[col].nunique() > 10]
    
    # Return two DataFrames: one for categorical (to one-hot encode) and one for continuous features
    return df_non_binary[categorical_cols], df_non_binary[continuous_cols]


#def nn_model(hidden_neurons_num):
    #input_dim


def centering(panda):
    float_columns=choose_floatCols(panda)
    float_numPy=float_columns.to_numpy()
    normalizer_layer=tf.keras.layers.Normalization(axis=-1)
    normalizer_layer.adapt(float_numPy)

    mean=normalizer_layer.mean
    
    return float_numPy - mean.numpy()

def min_max(panda):
    float_columns=choose_floatCols(panda)
    float_numPy=float_columns.to_numpy()

    min=float_numPy.min().to_numpy()
    max=float_numPy.max().to_numpy()

    scale = 1 / (max - min) 
    offset = -min * scale 

    scaler=tf.keras.layers.Lambda(lambda x: x*scale + offset)

    return scaler(float_numPy)

def z_score(panda,bin_cols):
    df = pd.concat([panda, bin_cols], axis=1)
    float_columns=choose_floatCols(df)
    float_numPy=float_columns.to_numpy()

    normalizer_layer=tf.keras.layers.Normalization(axis=-1)
    normalizer_layer.adapt(float_numPy)

    return normalizer_layer(float_numPy)

def min_max_scikit(panda):
    float_columns = choose_floatCols(panda)
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    normalized_data = scaler.fit_transform(float_columns)
    
    return normalized_data

def centering_scikit(panda):
    float_columns = choose_floatCols(panda)
    scaler = StandardScaler(with_std=False)  
    
    centered_data = scaler.fit_transform(float_columns)
    
    return centered_data

def z_score_scikit(panda,bin_cols):
    float_columns = choose_floatCols(panda)
    df = pd.concat([float_columns, bin_cols], axis=1)
    print("This is the dfQ")
    print(df)
    #float_columns=choose_floatCols(df)
    #loat_columns = choose_floatCols(panda)
    scaler = StandardScaler()  
    
    zscored_data = scaler.fit_transform(df)
    
    return zscored_data


def one_hot_encoding(panda):
    #columns = choose_nonBinCols(panda)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    encoder.fit(panda)
    encoded_columns = encoder.transform(panda)
    
    return encoded_columns


if __name__=='__main__':
    original_data=pd.read_csv("alzheimers_disease_data.csv")
    
    input=original_data.drop(["Diagnosis","PatientID","DoctorInCharge"], axis=1)
    print("THIS IS THE INPUT:")
    print(input.head())
    output=original_data["Diagnosis"].to_numpy()
    
    binary_input=choose_binCols(input).to_numpy()
    print(binary_input)
    print(binary_input.shape)       # This will print something like (num_samples, num_columns)
    print(binary_input.shape[1])

    remove_cols(original_data,["PatientID","DoctorInCharge"],True)

    #print(original_data.head())
    non_bin, bin_cols=choose_nonBinCols(input)
    print("--------------")
    print(non_bin.head())
    print("--------------")
    print(bin_cols)

    centered_data=centering_scikit(original_data)
    #print(centered_data)

    min_max_data=min_max_scikit(original_data)
    #print(min_max_data)

    z_data=z_score_scikit(original_data,bin_cols)
    print("this is the z data")
    print(z_data)
    print(z_data.shape[1]) 
    print("--------------")
    enc_data=one_hot_encoding(non_bin)
    print(enc_data)
    print(enc_data.shape)       
    print(enc_data.shape[1]) 

    filtered_input=np.concatenate([z_data,enc_data,binary_input], axis=1)
    print(filtered_input)
    print(filtered_input.shape)       
    print(filtered_input.shape[1]) 

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)










