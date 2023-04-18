"""
Authors: Kieran Ahn, Jaycee Nakagawa

Follows the procedure presented in https://github.com/botonobot/Understanding-Transformers-for-Bot-Detection-Twitter/blob/master/bot_detection/bot_detection-BERT_GPT2-transformers.ipynb
"""

import datetime
import argparse
import numpy as np
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel, BertTokenizer, BertForSequenceClassification
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.metrics import classification_report
import time

GPT_HIDDEN_DIM = 768
NUM_CLASSES = 2
BATCH_SIZE = 8
EPOCHS = 2
SEED = 2345
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"{datetime.datetime.now()}: Initializing GPT-2")

    if args.model == "gpt-2":
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True, pad_token='0', padding_side='right', truncation_side='right')
        model = GPT2DoubleHeadsModel.from_pretrained('gpt2', num_labels=NUM_CLASSES)
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_CLASSES)

    model.to(device)

    model_parameters = list(model.named_parameters())
    decay_free_parameters = ['bias', 'gamma', 'beta']
    parameters_for_optimizer = [
        {'params': [parameter for name, parameter in model_parameters if not any(no_decay in name for no_decay in decay_free_parameters)], 'weight_decay_rate':0.01},
        {'params': [parameter for name, parameter in model_parameters if any(no_decay in name for no_decay in decay_free_parameters)], 'weight_decay_rate':0.0},
    ]

    optimizer = AdamW(parameters_for_optimizer, lr=args.learning_rate)

    # Load the data.
    print(f"{datetime.datetime.now()}: Loading training data")
    train = pd.read_json("./Twibot-20/train.json")
    print(f"{datetime.datetime.now()}: Loading test data")
    test = pd.read_json("./Twibot-20/test.json")
    print(f"{datetime.datetime.now()}: Loading validation data")
    validation = pd.read_json("./Twibot-20/dev.json")

    if(args.subset_data):
        print("\nReducing data by factor of... a lot.")
        print(f'Train before: {train.shape[0]}')
        train = train[:1]
        print(f'Train after: {train.shape[0]}')
        
        print(f'Test before: {test.shape[0]}')
        test = test[:1]
        print(f'Test after: {test.shape[0]}')

        print(f'Validation before: {validation.shape[0]}')
        validation = validation[:1]
        print(f'Validation after: {validation.shape[0]}')

    print(f'\n{datetime.datetime.now()}: Making training dataset')
    train_tweets_labels = train[['tweet', 'label']].explode('tweet')
    train_tweets_labels = train_tweets_labels[~train_tweets_labels['tweet'].isnull()]

    if(args.subset_data):
        train_tweets_labels = train_tweets_labels[:104]
    
    print(f'Tokenizing data')
    if args.model == "gpt-2":
        train_tweets_masks = tokenizer(train_tweets_labels['tweet'].values.tolist(), padding=True, truncation=True, max_length=1024)
        train_tweets = torch.LongTensor(train_tweets_masks['input_ids'])
        train_masks = torch.FloatTensor(train_tweets_masks['attention_mask'])
        train_labels = torch.LongTensor(train_tweets_labels['label'].values)

    else:
        train_tweets = [tokenizer(tweet, truncation=True, max_length=2048, return_tensors='pt')['input_ids'].squeeze() for tweet in train_tweets_labels['tweet'].values]
        train_labels = train_tweets_labels['label'].values

    print(f"{datetime.datetime.now()}: Importing to TensorDataset")
    train_dataset = TensorDataset(train_tweets, train_masks, train_labels)

    print(f"{datetime.datetime.now()}: Making training dataloader")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    print(f"\n{datetime.datetime.now()}: Making validation dataset")
    val_tweets_labels = validation[['tweet','label']].explode('tweet')
    val_tweets_labels = val_tweets_labels[~val_tweets_labels['tweet'].isnull()]

    if(args.subset_data):
        val_tweets_labels = val_tweets_labels[:10]

    print(f'Tokenizing data')
    if args.model == "gpt-2":
        val_tweets_masks = tokenizer(val_tweets_labels['tweet'].values.tolist(), padding=True, truncation=True, max_length=1024)
        val_tweets = torch.LongTensor(val_tweets_masks['input_ids'])
        val_masks = torch.FloatTensor(val_tweets_masks['attention_mask'])
        val_labels = torch.LongTensor(val_tweets_labels['label'].values)
    else:
        val_tweets = [tokenizer(tweet, truncation=True, max_length=1024, return_tensors='pt')['input_ids'].squeeze() for tweet in val_tweets_labels['tweet'].values]
        val_labels = val_tweets_labels['label'].values

    print(f"{datetime.datetime.now()}: Importing to TensorDataset")
    val_dataset = TensorDataset(val_tweets, val_masks, val_labels)

    print(f"{datetime.datetime.now()}: Making validation dataloader")
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print(f"\n{datetime.datetime.now()}: Making test dataset")
    test_tweets_labels = test[['tweet','label']].explode('tweet')
    test_tweets_labels = test_tweets_labels[~test_tweets_labels['tweet'].isnull()]

    if(args.subset_data):
        test_tweets_labels = test_tweets_labels[:10]

    if args.model == "gpt-2":
        test_tweets_masks = tokenizer(test_tweets_labels['tweet'].values.tolist(), padding=True, truncation=True, max_length=1024)
        test_tweets = torch.LongTensor(test_tweets_masks['input_ids'])
        test_masks = torch.FloatTensor(test_tweets_masks['attention_mask'])
        test_labels = torch.LongTensor(test_tweets_labels['label'].values)
    else:
        test_tweets = [tokenizer(tweet, truncation=True, max_length=1024, return_tensors='pt')['input_ids'].squeeze() for tweet in test_tweets_labels['tweet'].values]
        test_labels = test_tweets_labels['label'].values

    # Train the model for the specified number of epochs.
    for epoch in range(EPOCHS):
        model.train()
        print(f'Epoch {epoch+1}')
        print('Training model...')
        train_iterator = iter(train_dataloader)
        avg_training_loss = 0

        batches = 0
        for tweets, masks, labels in tqdm(train_iterator):

            optimizer.zero_grad()

            if args.model == "gpt-2":
                loss = model(input_ids=tweets.to(device), token_type_ids=None, attention_mask=masks.to(device), mc_labels=labels.to(device)).mc_loss
            else:
                loss = model(input_ids=tweets.to(device), token_type_ids=None, attention_mask=masks.to(device), labels=labels.to(device)).mc_loss

            avg_training_loss += loss.item()
            batches += 1

            loss.backward()
            optimizer.step()
        
        avg_training_loss = avg_training_loss / batches
        print(f'Average training loss: {avg_training_loss}\n')
        
        print("Validating model...")
        with torch.no_grad():
            model.eval()

            validation_accuracy = 0

            for tweets, masks, labels in tqdm(iter(val_dataloader)):
                if args.model == "gpt-2":
                    logits = model(input_ids=tweets.to(device), token_type_ids=None, attention_mask=masks.to(device), mc_labels=labels.to(device)).mc_logits
                else:
                    logits = model(input_ids=tweets.to(device), token_type_ids=None, attention_mask=masks.to(device), labels=labels.to(device)).logits

                logits = logits.detach().cpu().numpy()
                validation_accuracy += np.sum(np.argmax(logits, axis=1).flatten() == labels.numpy().flatten())

            validation_accuracy = validation_accuracy / len(val_labels)
            print(f'Validation accuracy: {validation_accuracy}\n')
        
    print("Saving model...")
    torch.save(model, f'{args.model.upper()}/{args.model}_BOT_DETECTION.pt')

    # Evaluate the model.
    with torch.no_grad():
        torch.cuda.empty_cache()
        model.eval()
        print('\nTesting model...')

        if args.model == "gpt-2":
            logits = model(input_ids=test_tweets.to(device), token_type_ids=None, attention_mask=test_masks.to(device)).mc_logits
        else:
            logits = model(input_ids=test_tweets.to(device), token_type_ids=None, attention_mask=test_masks.to(device)).logits

        logits = logits.detach().cpu().numpy()
        
        predictions = np.argmax(logits, axis=1)
        
        report = classification_report(predictions, test_labels.numpy().flatten(), target_names=['human', 'bot'])

        print(f'Summary statistics for {args.model} bot detection network.')
        print(report)

        with open(f'{args.model}_report', 'wt') as model_report:
            model_report.write(report)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for gradient descent.')
    parser.add_argument('--subset_data', type=bool, default=False, help="Whether to use a subset of the total data (smaller by factor of 10) for faster training.")
    parser.add_argument('--model', default='gpt-2', choices=['bert', 'gpt-2'])

    args = parser.parse_args()

    main(args)
