import pymongo, pprint, pickle
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


def make_mask(X, mask: np.ndarray) -> np.ndarray:
        return X * mask.reshape(1, -1).astype(np.float32)


def load_data_and_model():
   
   with open("../best_split.pkl", "rb") as f:
    (X_train, y_train), (X_val, Y_val) = pickle.load(f)

    model = tf.keras.models.load_model("../best_model.keras")
    
    return X_train, y_train,X_val,Y_val,model


def build_model(input_dim):

    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Track validation loss
    patience=10,
    min_delta=0.001,     # Require at least 0.001 improvement        
    restore_best_weights=True  # Revert to best model weights
    )
   
    l1 = tf.keras.regularizers.l1(0.001) 

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l1),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l1),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.6)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model,early_stopping

#do a 5-cv fold split only on the training data only with the columns selected by the genetic algorithm(the code is from part 1)
def normal_training(filtered_input,output):
        file_fold_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)
        round=1
        evals=[]
        val_loss_table=[]
        loss_table=[]
        early_stop_epochs=[]
        batch_size=32

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath='../best_model_geneticv2.keras',      
            monitor='val_accuracy',         
            mode='max',                       
            save_best_only=True,              
            verbose=1
        )

        #for every split train the nn and evaluate it 
        for training_idx,val_idx in file_fold_split.split(filtered_input,output):

            input_train,input_val=filtered_input[training_idx],filtered_input[val_idx]
            output_train,output_val=output[training_idx],output[val_idx]

            model,early_stop=build_model(filtered_input.shape[1])
            training=model.fit(input_train, output_train,validation_data=(input_val, output_val),epochs=1100, batch_size=batch_size, verbose=1,callbacks=[early_stop,checkpoint])

            stop_epoch=len(training.history['loss'])
            early_stop_epochs.append(stop_epoch)
 
            val_loss_table.append(training.history['val_loss'])
            loss_table.append(training.history['loss'])


            evaluation=model.evaluate(input_val,output_val,verbose=0)
            print(f"Round {round}: Loss:{evaluation[0]}, Accuracy:{evaluation[1]}")
            round+=1
            evals.append(evaluation)

        max_epochs = max(early_stop_epochs)
        plot(loss_table,val_loss_table,max_epochs)


        
        #write the results to mongodb for further analysis
        evals_np=np.array(evals)
        evals_json={
            "Average loss": np.mean(evals_np[:, 0]),
            "Average Accuracy": np.mean(evals_np[:, 1])
        }

        printer=pprint.PrettyPrinter(indent=4)
        print('\n')
        print("|--------FINAL RESULTS----------|")
        printer.pprint(evals_json)


def plot(loss_table,val_loss_table,max_epochs):
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
        filename="genetic_nn.png"
        plt.savefig(filename,format='png')


def main(row: int=7):
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    col = client['genetic_algos']['resultsv2']
    tag_prefix = f"row{row}_run"

    docs = list(col.find({"params.tag": {"$regex": f"^{tag_prefix}"}}))

    #test only with the selected columns
    best_doc = max(docs, key=lambda d: d['best_fitness'])
    best_mask = np.array(best_doc['best_mask'], dtype=np.int8) 
    print(f"Selected best run: {best_doc['params']['tag']} with fitness {best_doc['best_fitness']}")

    X_tr,Y_tr,X_val,Y_val,old_model=load_data_and_model()

    #retraining
    X_mask=make_mask(X_val,best_mask)
    genetic_loss,genetic_acc,genetic_mse=old_model.evaluate(X_mask,Y_val, verbose=0)
    loss,acc,mse=old_model.evaluate(X_val,Y_val, verbose=0)

    print(f"Genetic results: loss:{genetic_loss}, Accuracy:{genetic_acc}, mse:{genetic_mse}")
    print(f"Old results: loss:{loss}, Accuracy:{acc}, mse:{mse}")


    selected_columns = np.where(best_mask == 1)[0]
    X_new = X_tr[:, selected_columns]
    print(f"Number of features selected: {X_new.shape[1]}")

    normal_training(X_new,Y_tr)



if __name__ == "__main__":
    main(row=6)