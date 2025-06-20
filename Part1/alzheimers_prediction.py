import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse, pickle
from pymongo import MongoClient
import os, pprint
from datetime import datetime

client = MongoClient("mongodb://localhost:27017/")
database=client["alzheimers"]
average_results=database["average_results"]
L1_collection=database["deep_L1"]
L2_collection=database["deep_L2"]

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------Data Pre-Processing---------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#

#some actions based on the columns of the datase

def choose_binCols(panda):
    binary_columns=[col for col in panda.columns if panda[col].nunique() == 2]

    return binary_columns

#find the categorical data for one hot encoding and the continuous data for centering/min-max/z-score
def seperate_columns(panda):
    binary_columns=choose_binCols(panda)
    non_bin_columns = panda.loc[:, ~panda.columns.isin(binary_columns)]

    categorical_cols = [col for col in non_bin_columns.columns if non_bin_columns[col].nunique() <= 10] #from checking the csv the categorical columns have less than 10 unique data
    continuous_cols = [col for col in non_bin_columns.columns if non_bin_columns[col].nunique() > 10]
    
    return categorical_cols,continuous_cols

#pre processing methods using scikit learn functions
def min_max(panda,columns):

    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    normalized_data = scaler.fit_transform(panda[columns])
    
    return normalized_data

def centering(panda,columns):
    scaler = StandardScaler(with_std=False)  
    
    centered_data = scaler.fit_transform(panda[columns])
    
    return centered_data

def z_score(panda,columns):
    scaler = StandardScaler()  
    
    zscored_data = scaler.fit_transform(panda[columns])
 
    return zscored_data


def one_hot_encoding(panda,columns):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    encoder.fit(panda[columns])
    encoded_data = encoder.transform(panda[columns])
    
    return encoded_data


#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------Neural Network--------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#

# the neural network architecture
def neural_network_model(input_shape,optimizer,momentum,lr,num_of_layers,hid_layer_func,loss_func,use_l2,use_l1,r=0.001,deep=False,deep_layers=None):
    
    hidden_layers={'half':math.ceil(input_shape/2), #diffrent choices for the neuron of the hidden layers all viable
                   "two thirds":math.ceil((2*input_shape)/3),
                   "same":input_shape,
                   "double":2*input_shape}
    
    activation_options={"Relu":"relu", #activation options of hidden layer
                        "Tanh":"tanh",
                        "Silu":tf.nn.silu}
    l=None
    if use_l2==False and use_l1 ==False:
       l=None
    elif use_l2 ==True:
        l=tf.keras.regularizers.L2(r) #L2 regulazation if you want 
    elif use_l1 == True:
        l=tf.keras.regularizers.L1(r)
    else:
        raise ValueError("Unsupported option")
     
    
    options_loss={"cross entropy":tf.keras.losses.BinaryCrossentropy(), #the loss functions
                  "MSE":tf.keras.losses.MeanSquaredError(name='MSE')}
    
    metrics={"Accuracy":'accuracy',
             "MSE":tf.keras.metrics.MeanSquaredError(name='MSE')} #the metrics of the nn

    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Track validation loss
    patience=10,
    min_delta=0.001,     # Require at least 0.001 improvement        
    restore_best_weights=True  # Revert to best model weights
)
    
    
    print(f"Running the model for:{hidden_layers[num_of_layers]} neurons")
    #the model itself, the number of output neurons is 1 because the patient has either alzheimers or not and using sigmoid as the activation champion we achieve the 
    #the binary clissification
    if deep ==False:
        model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(input_shape,)),
            tf.keras.layers.Dense(hidden_layers[num_of_layers], activation=activation_options[hid_layer_func],kernel_regularizer=l),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    else:
        print(f"Running for layers:{deep_layers}")
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(input_shape,)))
           
        for layer in deep_layers[1:]:
            model.add(tf.keras.layers.Dense(layer, activation=activation_options[hid_layer_func],kernel_regularizer=l))
        
    
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    
    #optimizer options
    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum,nesterov=True)
    else:
        raise ValueError("Unsupported option")
    
    
    model.compile(optimizer=optimizer, loss=options_loss[loss_func], metrics=[metrics["Accuracy"],metrics['MSE']])

    return model, early_stopping


#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------Parser----------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#



def create_parser():
    #argument parser to be able to test the training process with diffrent variables
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
    parser.add_argument("--test_lr_moment",type=bool,default=False,help="Test learning and momentum")
    parser.add_argument("--test_reg",type=bool,default=False,help="Test regularazation rate")
    parser.add_argument("--normal",type=bool,default=True,help="Normal training you pass ALL the paramaters")
    parser.add_argument("--compare_losses",type=bool,default=False,help="Toggle it to true if you want to see the evaluation losses seperatly")
    parser.add_argument("--more_layers",type=bool,default=False,help="Test the network with more layers")
    parser.add_argument("--use_l2",type=bool,default=False,help="Use L2")
    parser.add_argument("--use_l1",type=bool,default=False,help="Use L1")
    parser.add_argument('--hidden_layers', type=str, default="",
                        help="Comma-separated list of hidden layer sizes, e.g., '64,32' or '128,64,32'")

    args = parser.parse_args()


    return args


#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------Helper Functions------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#

#create a folder where the plots are stored  based on the variables of the run and the date
def create_folder(optimizer,momentum,lr,hidd_layers,hidd_func,loss_func,r,hidden_layers):
    date_str = datetime.now().strftime("%m-%d_%H-%M-%S")
    folder_name = f"screenshots/{optimizer}_mom{momentum}_lr{lr}_{hidd_layers}_{hidd_func}_{loss_func}_{r}_more_layers_{hidden_layers}{date_str}"

    os.makedirs(folder_name, exist_ok=True)

    return folder_name

#plot the taining-loss/validation loss 
def plot(args,loss_table,val_loss_table,folder,max_epochs):
        if args.compare_losses == True:
            for i, history in enumerate(val_loss_table):
                epochs = range(1, len(history) + 1)
                plt.plot(epochs, history, label=f'Fold {i+1}')
            
            plt.xlabel('Epoch')
            plt.ylabel('Validation Loss')
            plt.title('Validation Loss per Fold with Stopping Epochs')
            plt.legend()
            filename = os.path.join(folder, f"Plot.png")
            plt.savefig(filename,format='png')
            plt.show()
        else:
            # Determine the maximum number of epochs (or use args.epochs)
           
            print(f"This is the max epochs:{max_epochs}")

            # Pad each fold's validation loss history if it stopped early
            padded_val_histories = []
            padded_train_histories = []
            for train_history,val_history in zip(loss_table,val_loss_table):
                if len(train_history) < max_epochs:
                    pad_length = max_epochs - len(train_history)
                    padded_train = train_history + [train_history[-1]] * pad_length
                else:
                    padded_train = train_history[:max_epochs]
                
                if len(val_history) < max_epochs:
                    pad_length = max_epochs - len(val_history)
                    padded_val = val_history + [val_history[-1]] * pad_length
                else:
                    padded_val = val_history[:max_epochs]
                
                padded_train_histories.append(padded_train)
                padded_val_histories.append(padded_val)
            
            # Compute the average validation loss at each epoch across folds
            avg_loss = np.mean(padded_train_histories, axis=0)
            avg_val_loss = np.mean(padded_val_histories, axis=0)
            epochs = range(1, len(avg_loss) + 1)
            
            # Plot the average validation loss curve and save it in the created folder
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, avg_val_loss, label='Average Validation Loss', color='green')
            plt.plot(epochs, avg_loss, label='Average Training Loss', color='red')
            plt.xlabel('Epoch')
            plt.ylabel('Validation Loss')
            plt.title('Average Validation and Training Loss Over 5-Fold CV')
            plt.legend()
            filename = os.path.join(folder, f"Plot.png")
            plt.savefig(filename,format='png')


#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------Functions that run for the paramters asked without early stopping--------------------------------------------------------------------#
#--------------------------To see the more unstable variables---------------------------------------------------------------------------------------------------#


def run_for_many_layers(input_shape,filtered_input,output,args):
    hidden_layers={'half':math.ceil(input_shape/2), #diffrent choices for the neuron of the hidden layers all viable
                   "two thirds":math.ceil((2*input_shape)/3),
                   "same":input_shape,
                   "double":2*input_shape}
    
    val_loss_table=np.zeros((args.epochs, len(hidden_layers)))
    
    for i,(layer,value) in enumerate(hidden_layers.items()):
        file_fold_split =  StratifiedKFold(n_splits=5, shuffle=True, random_state=44)#5-cv fold with balanced output class data(StatifiedKFold does that)
        round=1
        for training_idx,val_idx in file_fold_split.split(filtered_input,output):
            print(f"Layer:{layer}, Value:{value},Round:{round}")

            input_train,input_val=filtered_input[training_idx],filtered_input[val_idx]
            output_train,output_val=output[training_idx],output[val_idx]

            model,_=neural_network_model(filtered_input.shape[1],args.optimizer,args.momentum,args.lr,layer,args.hid_layer_func,args.loss_func,False,False,args.r,False,None)
            training=model.fit(input_train, output_train,validation_data=(input_val, output_val) ,epochs=args.epochs, batch_size=32, verbose=1)

            val_loss_table[:,i] += training.history['val_loss']
            round+=1

    val_loss_table /= 5
    
    plt.plot(val_loss_table, label=hidden_layers.keys())
    plt.title(f"Average validation loss for each number of hidden layers")
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross Entropy')
    plt.legend()
    plt.show()

def test_lr_and_moment(filtered_input,output,args):
    testers=[(0.001,0.2),(0.001,0.6),(0.05,0.6),(0.1,0.6)]

    val_loss_table=np.zeros((args.epochs, len(testers)))
    
    for i,_ in enumerate(testers):
        file_fold_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=44) #5-cv fold with balanced output class data(StatifiedKFold does that)

        for training_idx,val_idx in file_fold_split.split(filtered_input,output):

            input_train,input_val=filtered_input[training_idx],filtered_input[val_idx]
            output_train,output_val=output[training_idx],output[val_idx]

            model,_=neural_network_model(filtered_input.shape[1],args.optimizer,testers[i][1],testers[i][0],args.num_of_layers,args.hid_layer_func,args.loss_func,False,False,args.r,False,None)
            training=model.fit(input_train, output_train,validation_data=(input_val, output_val) ,epochs=args.epochs, batch_size=128, verbose=1)

            val_loss_table[:,i] += training.history['val_loss']

    val_loss_table /= 5

    plt.plot(val_loss_table, label=list(map(lambda x: f"h={x[0]}, m={x[1]}", testers)))
    plt.title(f"Average validation loss for each number of learning rates and momentums")
    
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross Entropy')
    plt.legend()
    plt.show()



def test_reg(filtered_input,output,args):
    testers=[0.0001,0.001,0.01]

    val_loss_table=np.zeros((args.epochs, len(testers)))
    
    for i,_ in enumerate(testers):
        file_fold_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=44) #5-cv fold with balanced output class data(StatifiedKFold does that)

        for training_idx,val_idx in file_fold_split.split(filtered_input,output):

            input_train,input_val=filtered_input[training_idx],filtered_input[val_idx]
            output_train,output_val=output[training_idx],output[val_idx]

            
            model,_=neural_network_model(filtered_input.shape[1],args.optimizer,args.momentum,args.lr,args.num_of_layers,args.hid_layer_func,args.loss_func,args.use_l2,args.use_l1,testers[i],False,None)
            training=model.fit(input_train, output_train,validation_data=(input_val, output_val) ,epochs=args.epochs, batch_size=32, verbose=1)
            
            val_loss_table[:,i] += training.history['val_loss']
             

    val_loss_table /= 5

    plt.plot(val_loss_table, label=list(map(lambda x: f"r={x}", testers)))
    plt.title(f"Average validation loss for each number of rates")
    
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross Entropy')
    plt.legend()
    plt.show()



#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------Test if the split is correct------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#



def test_split():
    args=create_parser()

    original_data=pd.read_csv("alzheimers_disease_data.csv")
    input=original_data.drop(["Diagnosis","PatientID","DoctorInCharge"], axis=1)
    output=original_data["Diagnosis"].to_numpy()
    print(output)

    categorical_cols,contin_cols=seperate_columns(input)
    bin_cols=choose_binCols(input) #binary columns dont need any pre-processing

    #pre-process the columns, the continuous data is pre-processed with these 3 functions
    if args.pre_processing == "centering":
        pre_processed_input=centering(input,contin_cols)
    elif args.pre_processing == "z-score":
        pre_processed_input=z_score(input,contin_cols)
    elif args.pre_processing == "min-max":
        pre_processed_input=min_max(input,contin_cols)
    else:
        raise ValueError("Unsupported option")
    
    encoded_input=one_hot_encoding(input,categorical_cols) #the categorical data are being one hot encoded
    binary_input=input[bin_cols].to_numpy()

    filtered_input=np.concatenate([pre_processed_input,encoded_input,binary_input], axis=1) #concatenate the input to do 5-fold on them and put the nn 
    print(filtered_input)
    print(filtered_input.shape)       
    print(filtered_input.shape[1]) 

    file_fold_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=44) #5-cv fold with balanced output class data(StatifiedKFold does that)


    #for every split train the nn and evaluate it 
    for fold, (train_idx, val_idx) in enumerate(file_fold_split.split(filtered_input, output), start=1):
        train_classes, train_counts = np.unique(output[train_idx], return_counts=True)
        val_classes, val_counts = np.unique(output[val_idx], return_counts=True)
        
        print(f"Fold {fold}:")
        print("Training set class distribution:", dict(zip(train_classes, train_counts)))
        print("Validation set class distribution:", dict(zip(val_classes, val_counts)))
        print("-" * 30)


#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------The main code---------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#

def normal_training(filtered_input,output,args,folder,hidden_layers):
        file_fold_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=44) #5-cv fold with balanced output class data(StatifiedKFold does that)
        round=1
        evals=[]
        val_loss_table=[]
        loss_table=[]
        early_stop_epochs=[]
        io_vector=[]
        batch_size=32

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath='../best_model.keras',      
            monitor='val_accuracy',         
            mode='max',                       
            save_best_only=True,              
            verbose=1
        )

        #for every split train the nn and evaluate it 
        for training_idx,val_idx in file_fold_split.split(filtered_input,output):

            input_train,input_val=filtered_input[training_idx],filtered_input[val_idx]
            output_train,output_val=output[training_idx],output[val_idx]

            model,early_stop=neural_network_model(filtered_input.shape[1],args.optimizer,args.momentum,args.lr,args.num_of_layers,args.hid_layer_func,args.loss_func,args.use_l2,args.use_l1,args.r,args.more_layers,hidden_layers)
            training=model.fit(input_train, output_train,validation_data=(input_val, output_val),epochs=args.epochs, batch_size=batch_size, verbose=1,callbacks=[early_stop,checkpoint])

            stop_epoch=len(training.history['loss'])
            early_stop_epochs.append(stop_epoch)
 
            val_loss_table.append(training.history['val_loss'])
            loss_table.append(training.history['loss'])


            evaluation=model.evaluate(input_val,output_val,verbose=0)
            io_vector.append(((input_train, output_train),(input_val, output_val)))
            print(f"Round {round}: Loss:{evaluation[0]}, Accuracy:{evaluation[1]}")
            round+=1
            evals.append(evaluation)

        max_epochs = max(early_stop_epochs)
        plot(args,loss_table,val_loss_table,folder,max_epochs)


        best_idx = max(range(len(evals)),
               key=lambda i: evals[i][1])

        best_evaluation = evals[best_idx]
        print("This is the best eval:",best_evaluation)
        train_idx, val_idx = io_vector[best_idx]

        with open("../best_split.pkl", "wb") as f:
            pickle.dump((train_idx, val_idx), f)
        print(f"Saved best split (fold {best_idx+1}) → best_split.pkl")

        
        #write the results to mongodb for further analysis
        evals_np=np.array(evals)
        evals_json={
            "use L2":args.use_l2,
            "use L1":args.use_l1,
            "multiple layers":args.more_layers, #ignore for 1 layer
            "choosen architecture":args.hidden_layers, #ignore for 1 layer
            "chosen_weight":"double",
            "params":{
                "pre_processing":args.pre_processing,
                "optimizer":args.optimizer,
                "momentum":args.momentum,
                "learning rate":args.lr,
                "epochs":args.epochs,
                "run_epochs":max_epochs, #the epochs of the training, it can be less thean epochs because i have early stop
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

        if args.more_layers == False:
            average_results.insert_one(evals_json)
        elif args.use_l1 == True:
            L1_collection.insert_one(evals_json)
        elif args.use_l2 == True:
            L2_collection.insert_one(evals_json)



def main():
    args=create_parser() 

    print(f"Raw hidden_layers argument: '{args.hidden_layers}' (type: {type(args.hidden_layers)})")

    hidden_layers = []
    if args.hidden_layers.strip():  # Check for non-empty string after stripping whitespace
        try:
            hidden_layers = [int(n.strip()) for n in args.hidden_layers.split(',') if n.strip()]
            if not hidden_layers:
                print("Warning: --hidden_layers contained only commas/whitespace, using no hidden layers")
            else:
                print(f"Using hidden layers: {hidden_layers}")
        except ValueError as e:
            print(f"Error: Invalid layer sizes in --hidden_layers '{args.hidden_layers}'. Use comma-separated integers.")
            raise
    else:
        print("ℹ No hidden layers specified - using default architecture")

    folder=create_folder(args.optimizer,args.momentum,args.lr,args.num_of_layers,args.hid_layer_func,args.loss_func,args.r,hidden_layers) #folder to save the plot

        
    original_data=pd.read_csv("../alzheimers_disease_data.csv")
    input=original_data.drop(["Diagnosis","PatientID","DoctorInCharge"], axis=1)
    
    output=original_data["Diagnosis"].to_numpy()

    categorical_cols,contin_cols=seperate_columns(input)
    bin_cols=choose_binCols(input) #binary columns dont need any pre-processing

    #pre-process the columns, the continuous data is pre-processed with these 3 functions
    if args.pre_processing == "centering":
        pre_processed_input=centering(input,contin_cols)
    elif args.pre_processing == "z-score":
        pre_processed_input=z_score(input,contin_cols)
    elif args.pre_processing == "min-max":
        pre_processed_input=min_max(input,contin_cols)
    else:
        raise ValueError("Unsupported option")
    
    encoded_input=one_hot_encoding(input,categorical_cols).astype(np.float32) #the categorical data are being one hot encoded
    binary_input=input[bin_cols].to_numpy().astype(np.float32)
    pre_processed_input = pre_processed_input.astype(np.float32)

    filtered_input=np.concatenate([pre_processed_input,encoded_input,binary_input], axis=1).astype(np.float32) #concatenate the input to do 5-fold on them and put the nn, flaot32 because it helps the gpu

    if args.all_weights == True:
        run_for_many_layers(filtered_input.shape[1],filtered_input,output,args)
    elif args.test_lr_moment == True:
        test_lr_and_moment(filtered_input,output,args)
    elif args.test_reg == True:
        test_reg(filtered_input,output,args)
    elif args.normal == True:
        normal_training(filtered_input,output,args,folder,hidden_layers)




if __name__=='__main__':

    main()
    #test_split() #put main in comments to see if the splitting is correct








