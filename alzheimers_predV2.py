import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras import regularizers
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
from pymongo import MongoClient
import os, pprint
from datetime import datetime

client = MongoClient("mongodb://localhost:27017/")
database=client["alzheimers"]
average_results=database["average_results"]

#some actions based on the columsn of the dataset
def remove_cols(panda,columns,inplace):
    return panda.drop(columns=columns,inplace=inplace)

def choose_cols(panda,columns):
    return panda[columns].copy()

def choose_intCols(panda):
    return panda.select_dtypes(include=["int64"])

def choose_floatCols(panda):
    return panda.select_dtypes(include=["float64"])

def choose_binCols(panda):
    binary_columns=[col for col in panda.columns if panda[col].nunique() == 2]

    return binary_columns

#find the categorical data for one hot encoding and the continuous data for centering/min-max/z-score
def seperate_columns(panda):
    binary_columns=choose_binCols(panda)
    non_bin_columns = panda.loc[:, ~panda.columns.isin(binary_columns)]

    categorical_cols = [col for col in non_bin_columns.columns if non_bin_columns[col].nunique() <= 10]
    continuous_cols = [col for col in non_bin_columns.columns if non_bin_columns[col].nunique() > 10]
    
    return categorical_cols,continuous_cols

#pre processing methods using scikit learn functions
def min_max_scikit(panda,columns):
    float_columns = choose_floatCols(panda[columns])
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    normalized_data = scaler.fit_transform(float_columns)
    
    return normalized_data

def centering_scikit(panda,columns):
    scaler = StandardScaler(with_std=False)  
    
    centered_data = scaler.fit_transform(panda[columns])
    
    return centered_data

def z_score_scikit(panda,columns):
    scaler = StandardScaler()  
    
    zscored_data = scaler.fit_transform(panda[columns])
    
    return zscored_data


def one_hot_encoding(panda,columns):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    encoder.fit(panda[columns])
    encoded_columns = encoder.transform(panda[columns])
    
    return encoded_columns

# the neural network architecture
def nn_model(input_shape,optimizer,momentum,lr,num_of_layers,hid_layer_func,loss_func,r):
    
    hidden_layers={'half':math.ceil(input_shape/2), #diffrent choices for the neuron of the hidden layers all viable
                   "two thirds":math.ceil((2*input_shape)/3),
                   "same":input_shape,
                   "double":2*input_shape}
    
    activation_options={"Relu":"relu", #activation options of hidden layer
                        "Tanh":"tanh",
                        "Silu":tf.nn.silu}
    l2=None
    if r==None:
       l2=None
    elif r is not None:
        l2=regularizers.L2(r) #L2 regulazation if you want 
    else:
        raise ValueError("Unsupported option")
     
    
    options_loss={"cross entropy":tf.keras.losses.BinaryCrossentropy(), #the loss functions
                  "MSE":tf.keras.losses.MeanSquaredError(name='MSE')}
    
    metrics={"Accuracy":'accuracy',
             "MSE":tf.keras.metrics.MeanSquaredError(name='MSE')} #the metrics of the nn

    early_stopping=tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        min_delta=0.0005,
        restore_best_weights=True #early stopping to avoid overfitting 
    )
    
    #the model itself, the number of output neurons is 1 because the patient has either alzheimers or not and using sigmoid as the activation champion we achieve the 
    #the binary clissification
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(hidden_layers[num_of_layers], activation=activation_options[hid_layer_func],kernel_regularizer=l2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    #optimizer options
    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    else:
        raise ValueError("Unsupported option")
    
    
    model.compile(optimizer=optimizer, loss=options_loss[loss_func], metrics=[metrics["Accuracy"],metrics['MSE']])

    return model, early_stopping

def create_parser():
    #argumetn parser to be able to test the training process with diffrent variables
    parser=argparse.ArgumentParser(description="options for Neural Network")

    parser.add_argument("--pre_processing",type=str,default="z-score",help="What pre processing function do you want to use options:centering,z-score,min-max")
    parser.add_argument("--optimizer",type=str,default="adam",help="Type the optimizer for weights you want to use options:adam,SGD")
    parser.add_argument("--lr",type=float,default=0.001,help="The learning rate for training")
    parser.add_argument("--momentum",type=float,default=0.0,help="THe momentum for the SGD")
    parser.add_argument("--epochs",type=int,default=50,help="Number of epochs to train")
    parser.add_argument("--num_of_layers",type=str,default="two thirds",help="The number of hidden layers options: I/2,2*I/3,I,2*I")
    parser.add_argument("--loss_func",type=str,default="cross entropy",help="The loss function options: cross entropy,MSE")
    parser.add_argument("--hid_layer_func",type=str,default="Relu",help="Activation function for hidden layers options:Relu,Tanh,Silu")
    parser.add_argument("--r",type=float,default=None,help="Regulazation factor")
    parser.add_argument("--all_weights",type=bool,default=False,help="Specify if you want to see the diifrent val loss bettwen all possible wights or to do normal training")

    args = parser.parse_args()

    return args

#create a folder where the plots are stored  based on the variables of the run and the date
def create_folder(optimizer,momentum,lr,hidd_layers,hidd_func,loss_func,r):
    date_str = datetime.now().strftime("%m-%d_%H-%M-%S")
    folder_name = f"screenshots/{optimizer}_mom{momentum}_lr{lr}_{hidd_layers}_{hidd_func}_{loss_func}_{r}_{date_str}"

    os.makedirs(folder_name, exist_ok=True)

    return folder_name

#plot the taining-loss/validation loss 
def plot(round,training_loss,val_loss,folder):
    plt.plot(training_loss,label=['Training CE'])
    plt.plot(val_loss,label=['Validation CE'])
    plt.title(f"Round {round}")
    
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross Entropy')
    plt.legend()

    filename = os.path.join(folder, f"Round{round}.png")
    plt.savefig(filename,format='png')
    plt.close()


def run_for_many_layers(input_shape,filtered_input,output,args):
    hidden_layers={'half':math.ceil(input_shape/2), #diffrent choices for the neuron of the hidden layers all viable
                   "two thirds":math.ceil((2*input_shape)/3),
                   "same":input_shape,
                   "double":2*input_shape}
    
    val_loss_table=np.zeros((args.epochs, len(hidden_layers)))
    
    for i,num_Layer in enumerate(hidden_layers):
        five_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=44) #5-cv fold with balanced output class data(StatifiedKFold does that)

        for training_idx,val_idx in five_fold.split(filtered_input,output):

            input_train,input_val=filtered_input[training_idx],filtered_input[val_idx]
            output_train,output_val=output[training_idx],output[val_idx]

            model,_=nn_model(filtered_input.shape[1],args.optimizer,args.momentum,args.lr,args.num_of_layers,args.hid_layer_func,args.loss_func,args.r)
            training=model.fit(input_train, output_train,validation_data=(input_val, output_val) ,epochs=args.epochs, batch_size=32, verbose=1)

            val_loss_table[:,i] += training.history['val_loss']

    val_loss_table /= 5
    
    plt.plot(val_loss_table, label=hidden_layers.keys())
    plt.title(f"Average validation loss for each number of hiiden layers")
    
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross Entropy')
    plt.legend()
    plt.show()

def test_lr_and_moment(filtered_input,output,args):
    testers=[(0.001,0.2),(0.001,0.6),(0.05,0.6),(0.1,0.6)]

    val_loss_table=np.zeros((args.epochs, len(testers)))
    
    for i,_ in enumerate(testers):
        five_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=44) #5-cv fold with balanced output class data(StatifiedKFold does that)

        for training_idx,val_idx in five_fold.split(filtered_input,output):

            input_train,input_val=filtered_input[training_idx],filtered_input[val_idx]
            output_train,output_val=output[training_idx],output[val_idx]

            model,_=nn_model(filtered_input.shape[1],args.optimizer,testers[i][0],testers[i][1],args.num_of_layers,args.hid_layer_func,args.loss_func,args.r)
            training=model.fit(input_train, output_train,validation_data=(input_val, output_val) ,epochs=args.epochs, batch_size=32, verbose=1)

            val_loss_table[:,i] += training.history['val_loss']

    val_loss_table /= 5

    plt.plot(val_loss_table, label=list(map(lambda x: f"h={x[0]}, m={x[1]}", testers)))
    plt.title(f"Average validation loss for each number of hiiden layers")
    
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross Entropy')
    plt.legend()
    plt.show()



def test_reg(filtered_input,output,args):
    testers=[0.0001,0.001,0.01]

    val_loss_table=np.zeros((args.epochs, len(testers)))
    
    for i,_ in enumerate(testers):
        five_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=44) #5-cv fold with balanced output class data(StatifiedKFold does that)

        for training_idx,val_idx in five_fold.split(filtered_input,output):

            input_train,input_val=filtered_input[training_idx],filtered_input[val_idx]
            output_train,output_val=output[training_idx],output[val_idx]

            
            model,_=nn_model(filtered_input.shape[1],args.optimizer,args.momentum,args.lr,args.num_of_layers,args.hid_layer_func,args.loss_func,testers[i])
            training=model.fit(input_train, output_train,validation_data=(input_val, output_val) ,epochs=args.epochs, batch_size=32, verbose=1)

            val_loss_table[:,i] += training.history['val_loss']

    val_loss_table /= 5

    plt.plot(val_loss_table, label=list(map(lambda x: f"r={x}", testers)))
    plt.title(f"Average validation loss for each number of hiiden layers")
    
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross Entropy')
    plt.legend()
    plt.show()
    


def main():
    args=create_parser() 
    folder=create_folder(args.optimizer,args.momentum,args.lr,args.num_of_layers,args.hid_layer_func,args.loss_func,args.r)
    
    original_data=pd.read_csv("alzheimers_disease_data.csv")
    input=original_data.drop(["Diagnosis","PatientID","DoctorInCharge"], axis=1)
    output=original_data["Diagnosis"].to_numpy()

    categorical_cols,contin_cols=seperate_columns(input)
    bin_cols=choose_binCols(input) #binary columns dont need any pre-processing

    #pre-process the columns, the continuous data is pre-processed with these 3 functions
    if args.pre_processing == "centering":
        pre_processed_input=centering_scikit(input,contin_cols)
    elif args.pre_processing == "z-score":
        pre_processed_input=z_score_scikit(input,contin_cols)
    elif args.pre_processing == "min-max":
        pre_processed_input=min_max_scikit(input,contin_cols)
    else:
        raise ValueError("Unsupported option")

    encoded_input=one_hot_encoding(input,categorical_cols) #the categorical data are being one hot encoded
    binary_input=input[bin_cols].to_numpy()

    filtered_input=np.concatenate([pre_processed_input,encoded_input,binary_input], axis=1) #concatenate the input to do 5-fold on them and put the nn 
    print(filtered_input)
    print(filtered_input.shape)       
    print(filtered_input.shape[1]) 

    if args.all_weights == True:
        run_for_many_layers(filtered_input.shape[1],filtered_input,output,args)
    else:
        five_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=44) #5-cv fold with balanced output class data(StatifiedKFold does that)
        round=1
        evals=[]
        val_json={}

        #for every split train the nn and evaluate it 
        for training_idx,val_idx in five_fold.split(filtered_input,output):

            input_train,input_val=filtered_input[training_idx],filtered_input[val_idx]
            output_train,output_val=output[training_idx],output[val_idx]

            model,early_stop=nn_model(filtered_input.shape[1],args.optimizer,args.momentum,args.lr,args.num_of_layers,args.hid_layer_func,args.loss_func,args.r)
            training=model.fit(input_train, output_train,validation_data=(input_val, output_val) ,epochs=args.epochs, batch_size=32, verbose=1,callbacks=[early_stop])

            plot(round,training.history['loss'],training.history['val_loss'],folder)


            evaluation=model.evaluate(input_val,output_val,verbose=0)
            print(f"Round {round}: Loss:{evaluation[0]}, Accuracy:{evaluation[1]}")
            round+=1
            evals.append(evaluation)
        
        #write the results to mongodb for further analysis
        evals_np=np.array(evals)
        evals_json={
            "params":{
                "optimizer":args.optimizer,
                "momentum":args.momentum,
                "learning rate":args.lr,
                "epochs":args.epochs,
                "number of hidden layers":args.num_of_layers,
                "hidden layer activation function":args.hid_layer_func,
                "regulazation rate":args.r,
                "loss function":args.loss_func
            },
            "Average loss": np.mean(evals_np[:, 0]),
            "Average Accuracy": np.mean(evals_np[:, 1]),
            "Average MSE":   np.mean(evals_np[:, 2]),
        }

        printer=pprint.PrettyPrinter(indent=4)
        print('\n')
        print("|--------FINAL RESULTS----------|")
        printer.pprint(evals_json)
        average_results.insert_one(evals_json)
    #print("\nΜέσο Loss:", np.mean(evals_np[:, 0]))
    #print("Μέση Accuracy:", np.mean(evals_np[:, 1]))
    #rint("Μέσο MSE:", np.mean(evals_np[:,2]))

if __name__=='__main__':

    main()







