from TLA.Lang_Classify.model import get_model, BERT_Arch
import argparse
import torch
import pandas as pd
import numpy as np
from transformers import AutoModel, BertTokenizerFast
import pickle
from distutils.sysconfig import get_python_lib

def predict(val_text,model):
    try:
        if isinstance(pd.read_csv(val_text),pd.DataFrame) == True:
            val_text = np.array(pd.read_csv(val_text))
    except:
        if isinstance(val_text,str) == True:
            val_text = np.array([val_text])
        else:
            return "First Argument must be of string or numpy array DataType"

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(),
        max_length = 512,
        padding='max_length',
        truncation=True
    )

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    le = pickle.load(open(get_python_lib() + "/TLA/ang_Classify/models/encoder.pkl","rb"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.to(device)
        preds = model(val_seq.to(device), val_mask.to(device))
        preds = preds.detach().cpu().numpy()
        preds = np.argmax(preds, axis=1)
        preds = le.inverse_transform(preds)
        return preds[0]

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--predict', action='store', type=str,required=True)
    my_parser.add_argument('--weights', action='store', type=str)
    args = my_parser.parse_args()
    model = get_model(args.weights)
    prediction = predict(args.predict,model)
    print(prediction)