#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 23:36:13 2023

@author: vamsipusapati
"""

from DataProcessing import *
    
import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import f1_score
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm



import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

baseDataLoader = BaseDataLoader


model = None


seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)



def get_f1_score(preds, labels):
    preds = np.argmax(preds, axis=1).flatten()
    label = labels.flatten()
    return f1_score(label, preds, average = 'weighted')

"""
    The below function is used to get the accuracy of each labels
"""
def get_accuracy_perclass(preds, labels):
    label_inverse = {v: k for k, v in label_dict.items()}

    preds = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        accuracy = len(y_preds[y_preds==label])/ len(y_true)
        print(f'Class: {label_inverse[label]}')
        print(f'Accuracy:{len(y_preds[y_preds==label])}/{len(y_true)} = {accuracy}\n')

def add_label(data_frame, category):
    df = data_frame.copy()
    df["label"] = category
    return df


def evaluate(batch_dataloder):

    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in batch_dataloder:
        batch = tuple(bat.to(device) for bat in batch)
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        predicts = outputs[1]
        loss_val_total += loss.item()

        predicts = predicts.detach().cpu().numpy()
        labels= inputs['labels'].cpu().numpy()
        predictions.append(predicts)
        true_vals.append(labels)

    loss_val_avg = loss_val_total/len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals

def train_berthModel_for_goemotion():
    
    positiveSet_bert = ['Clean_text','admiration','desire','love','amusement','excitement',
               'optimism','approval','gratitude','pride','caring','joy','relief']
    negativeSet_bert = ['Clean_text','anger','disgust','nervousness','annoyance','embarrassment',
                'remorse','disappointment', 'fear', 'sadness', 'disapproval', 'grief']
    neutralSet_bert = ['Clean_text','confusion','curiosity','realization','surprise']
    goEmotion_loader = GoEmotionDataLoader()
    posTrainDF, posTestDF, posTrainDF_conflated, posTestDF_conflated = goEmotion_loader.getEmotionTestTrainDF( positiveSet_bert,"positive" )
    negTrainDF, negTestDF, negTrainDF_conflated, negTestDF_conflated = goEmotion_loader.getEmotionTestTrainDF( negativeSet_bert,"negative" )
    neuTrainDF, neuTestDF, neuTrainDF_conflated, neuTestDF_conflated = goEmotion_loader.getEmotionTestTrainDF( neutralSet_bert,"neutral" )
    
    train_pos_df = add_label(posTrainDF_conflated,"positive")
    train_neg_df = add_label(negTrainDF_conflated,"negative")
    train_neu_df = add_label(neuTrainDF_conflated,"neutral")

    test_pos_df = add_label(posTestDF_conflated,"positive")
    test_neg_df = add_label(negTestDF_conflated,"negative")
    test_neu_df = add_label(neuTestDF_conflated,"neutral")
    
    total_train_lenght = len(train_pos_df["Clean_text"]) +len(train_neg_df["Clean_text"])+len(train_neu_df["Clean_text"])
    ratio_of_pos = len(train_pos_df["Clean_text"])/total_train_lenght
    ratio_of_neg = len(train_neg_df["Clean_text"])/total_train_lenght
    ratio_of_neu = len(train_neu_df["Clean_text"])/total_train_lenght

    np.random.seed(1)
    
    #creating a combined dataframe for train
    train_df =pd.concat([train_pos_df, train_neg_df, train_neu_df], axis=0)
    # randomly shuffling the dataframe so that all the rows will be randomly mixed.
    train_df = train_df.sample(frac=1, random_state=1).reset_index(drop=True)
    
    #creating a combined dataframe for test no need shuffle test df as we are not using this to train our model
    test_df = pd.concat([test_pos_df, test_neg_df, test_neu_df], axis=0)
    
    
    
    label_dict = {
    "positive":0,
    "negative":1,
    "neutral":2
    }
    train_df.label = train_df["label"].map(label_dict)
    test_df.label = test_df["label"].map(label_dict)
        
    X_train, X_val, Y_train, Y_val = train_test_split(train_df["Clean_text"],
                                                  train_df["label"],
                                                  test_size=0.1,
                                                  random_state=34,
                                                  stratify=train_df["label"])

    
    test_text = test_df["Clean_text"]
    Y_test = test_df["label"]
    
    
    # using the pretrained tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encoding_texts = tokenizer.batch_encode_plus(X_train, padding="max_length", truncation=True, return_tensors="pt",)
    val_encoding_texts = tokenizer.batch_encode_plus(X_val, padding="max_length", truncation=True, return_tensors="pt",)

    train_input = train_encoding_texts['input_ids']
    train_attention = train_encoding_texts['attention_mask']
    train_labels = torch.tensor(Y_train.values)
    
    val_input_ids_val = val_encoding_texts['input_ids']
    val_attention_masks_val = val_encoding_texts['attention_mask']
    val_labels_val = torch.tensor(Y_val.values)
    
    train_tensor_data = TensorDataset(train_input, train_attention, train_labels)

    val_tensor_data = TensorDataset(val_input_ids_val, val_attention_masks_val, val_labels_val)
    
    
    model = BertForSequenceClassification.from_pretrained(
                                      'bert-base-uncased',
                                      num_labels = 3,
                                      output_attentions = True,
                                      output_hidden_states = False
                                     )
    
    
    
    batch_size = 16

    dataloader_train = DataLoader(
        train_tensor_data,
        sampler=RandomSampler(train_tensor_data),
        batch_size=batch_size
    )
    
    dataloader_val = DataLoader(
        val_tensor_data,
        sampler=RandomSampler(val_tensor_data),
        batch_size=batch_size
    )
        
    optimizer = AdamW(
    model.parameters(),
    lr = 1e-4,
    eps = 1e-6
    )
    
    
    
    epochs = 3

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps = len(dataloader_train)*epochs
    )
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)
    
    
    for epoch in tqdm(range(1, epochs+1)):
        model.train()
        total_train_loss = 0
    
        progress_bar = tqdm(dataloader_train,
                            desc='Epoch {:1d}'.format(epoch),
                            leave=False,
                            disable=False)
    
        for batch in progress_bar:
            model.zero_grad()
    
            batch = tuple(bat.to(device) for bat in batch)
    
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2]
            }
    
            outputs = model(**inputs)
            loss = outputs[0]
            total_train_loss +=loss.item()
            loss.backward()
    
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
            optimizer.step()
    
    
            scheduler.step()
    
            progress_bar.set_postfix({'training loss': '{:.4f}'.format(loss.item()/len(batch))})
    
    
    
        tqdm.write('\nEpoch {epoch}')
    
        avg_training_loss = total_train_loss/len(dataloader_train)
        tqdm.write(f'Training loss: {avg_training_loss}')
    
        val_loss, predictions, true_vals = evaluate(dataloader_val)
        val_f1 = get_f1_score(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (weighted): {val_f1}')
        
        
        get_accuracy_perclass(predictions, true_vals)
        bert_model = model
        return model
    
    
def get_goEmotions_test_predictions():
    positiveSet_bert = ['Clean_text','admiration','desire','love','amusement','excitement',
               'optimism','approval','gratitude','pride','caring','joy','relief']
    negativeSet_bert = ['Clean_text','anger','disgust','nervousness','annoyance','embarrassment',
                'remorse','disappointment', 'fear', 'sadness', 'disapproval', 'grief']
    neutralSet_bert = ['Clean_text','confusion','curiosity','realization','surprise']
    goEmotion_loader = GoEmotionDataLoader()
    posTrainDF, posTestDF, posTrainDF_conflated, posTestDF_conflated = goEmotion_loader.getEmotionTestTrainDF( positiveSet_bert,"positive" )
    negTrainDF, negTestDF, negTrainDF_conflated, negTestDF_conflated = goEmotion_loader.getEmotionTestTrainDF( negativeSet_bert,"negative" )
    neuTrainDF, neuTestDF, neuTrainDF_conflated, neuTestDF_conflated = goEmotion_loader.getEmotionTestTrainDF( neutralSet_bert,"neutral" )
    

    test_pos_df = add_label(posTestDF_conflated,"positive")
    test_neg_df = add_label(negTestDF_conflated,"negative")
    test_neu_df = add_label(neuTestDF_conflated,"neutral")
    

    np.random.seed(1)

    
    #creating a combined dataframe for test no need shuffle test df as we are not using this to train our model
    test_df = pd.concat([test_pos_df, test_neg_df, test_neu_df], axis=0)
    
    
    
    label_dict = {
    "positive":0,
    "negative":1,
    "neutral":2
    }

    test_df.label = test_df["label"].map(label_dict)
        
    test_text = test_df["Clean_text"]
    Y_test = test_df["label"]
    label_dict = {
        "positive":0,
        "negative":1,
        "neutral":2
    }
    Y_test = test_df["label"]
    Y_test.label = Y_test.map(label_dict)
    
    
    
    
    
    test_encoding_texts = tokenizer.batch_encode_plus(test_text, padding="max_length", truncation=True, return_tensors="pt",)
    test_input = test_encoding_texts['input_ids']
    test_attention = test_encoding_texts['attention_mask']
    test_labels = torch.tensor(Y_test.values)
    
    
    
    test_tensor_data = TensorDataset(test_input, test_attention, test_labels)
    
    
    batch_size = 16
    dataloader_test = DataLoader(
        test_tensor_data,
        sampler=RandomSampler(test_tensor_data),
        batch_size=batch_size
    )
    
    test_loss, test_predictions, test_true_vals = evaluate(dataloader_test)
    
    
    get_accuracy_perclass(test_predictions, test_true_vals)

    class_names = ["Positive", "Negative", "Neutral"]
    
    test_preds = np.argmax(test_predictions, axis=1).flatten()
    
    
    
    f1 = classification_report(test_true_vals, test_preds, target_names=[
                            "positive", "negative", "neutral"])


    print(f"\nF1 Score:\n{f1}")
    
    conf_matrix = confusion_matrix(test_true_vals, test_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig('test_confusion_matrix.png')
    plt.show()
    
    test_df["predict_label"] = test_preds
    rev_label_dict = {
        0: "positive",
        1: "negative",
        2: "neutral"
    }
    go_emotions_test_prediction = test_df.copy()
    
    go_emotions_test_prediction.label = go_emotions_test_prediction.label.map(rev_label_dict)
    go_emotions_test_prediction.predict_label = go_emotions_test_prediction.predict_label.map(rev_label_dict)
    
    return go_emotions_test_prediction
        