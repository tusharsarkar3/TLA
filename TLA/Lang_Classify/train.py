import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
import transformers
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModel, BertTokenizerFast
from transformers import AdamW
import argparse
from model import get_model, BERT_Arch
from training_utils import train,evaluate
import os
import pickle
import matplotlib.pyplot as plt

def train_lang(path,epochs,weights=None):
    directory = "models"
    parent_dir = "TLA\Lang_Classify"
    p = os.path.join(parent_dir, directory)
    if os.path.isdir(p) == False:
        os.mkdir(p)
    df = pd.read_csv(path)
    le = preprocessing.LabelEncoder()
    df.language = le.fit_transform(df.language)
    train_text, val_text, train_labels, val_labels = train_test_split(df['Text'], df['language'],
                                                                        random_state=0,
                                                                        test_size=0.3,
                                                                        stratify=df['language'])
    pickle.dump(le,open(p+"\encoder.pkl",'wb'))
    bert = AutoModel.from_pretrained('bert-base-uncased')

    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(),
        max_length = 512,
        padding='max_length',
        truncation=True
    )

    # tokenize and encode sequences in the validation set
    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(),
        max_length = 512,
        padding='max_length',
        truncation=True
    )

    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(val_labels.tolist())

    #define a batch size
    batch_size = 32

    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_y)

    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)

    # dataLoader for train set
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # wrap tensors
    val_data = TensorDataset(val_seq, val_mask, val_y)

    # sampler for sampling the data during training
    val_sampler = SequentialSampler(val_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataLoader for validation set
    val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

    model = get_model(weights)

    optimizer = AdamW(model.parameters(),
                      lr = 1e-5)

    # define the loss function
    cross_entropy  = nn.CrossEntropyLoss()

    if epochs == None:
        epochs = 10
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []

    # fo
    # r each epoch
    for epoch in range(epochs):

        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        # train model
        train_loss, _ = train(model,train_dataloader,optimizer,cross_entropy,device)

        # evaluate model
        valid_loss, _ = evaluate(model,val_dataloader,cross_entropy,device)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), p + '\saved_weights.pt')
            torch.save(model, p +'\saved_weights_full.pt')

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')

    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(1)
    figure.suptitle('Performance of TLA ')

    axis.plot(train_losses, label="Training Loss")
    axis.plot(valid_losses, label="Testing Loss")
    axis.set_xlabel('Epochs')
    axis.set_ylabel('Loss')
    axis.legend()

    plt.savefig('performance.png')
    plt.show()


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--data', action='store', type=str,required=True)
    my_parser.add_argument('--models', action='store', type=str)
    my_parser.add_argument('--epochs', action='store', type=int)
    args = my_parser.parse_args()
    train_lang(args.data,args.models)