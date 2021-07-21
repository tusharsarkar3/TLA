from transformers import AutoModel
import torch
import torch.nn as nn

class BERT_Arch(nn.Module):

    def __init__(self, bert):
        super(BERT_Arch, self).__init__()

        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768, 512)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, 22)

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)

        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)

        return x

def get_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert = AutoModel.from_pretrained('bert-base-uncased')
    for param in bert.parameters():
        param.requires_grad = False
    device = torch.device(device)
    model = BERT_Arch(bert)

    # push the model to GPU
    model = model.to(device)
    if path ==None:
        return model
    try:
        model.load_state_dict(torch.load(path))
    except:
        model = torch.load(path)
    finally:
        return model