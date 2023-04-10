"""
Author: Kieran Ahn

Follows the procedure presented in https://github.com/botonobot/Understanding-Transformers-for-Bot-Detection-Twitter/blob/master/bot_detection/bot_detection-BERT_GPT2-transformers.ipynb
"""

import datetime
import argparse
import numpy as np
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
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

    if args.model == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True)
        model = GPT2DoubleHeadsModel.from_pretrained('gpt2', num_labels=NUM_CLASSES)
    else:
        raise Exception("Bert is not supported yet!")

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
    print(f"{datetime.datetime.now()}: Loading validaton data")
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
    
    if args.model == "gpt2":
        train_tweets = [tokenizer(tweet, truncation=True, max_length=2048, return_tensors='pt')['input_ids'].squeeze().to(device) for tweet in train_tweets_labels['tweet'].values]
        train_labels = train_tweets_labels['label'].values
    else:
        raise Exception("Bert is not supported right now!")
    
    print(f'{datetime.datetime.now()}: Padding sequences')
    train_tweets = pad_sequence(train_tweets).to(device).transpose(0, 1)

    print(f'{datetime.datetime.now()}: Cloning tweet tensor')
    train_masks = train_tweets.detach().clone()
    print(f"{datetime.datetime.now()}: Creating training attention mask")
    train_masks = train_masks.cpu().apply_(lambda word: float(word>0)).to(device).float()

    print(f"{datetime.datetime.now()}: Tensorifying training dataset")
    train_tweets_tensor = train_tweets.long()
    train_masks_tensor = train_masks
    train_labels_tensor = torch.LongTensor(train_labels).to(device)

    print(f"{datetime.datetime.now()}: Importing to TensorDataset")
    train_dataset = TensorDataset(train_tweets_tensor, train_masks_tensor, train_labels_tensor)

    print(f"{datetime.datetime.now()}: Making training dataloader")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    print(f"\n{datetime.datetime.now()}: Making validation dataset")
    val_tweets_labels = validation[['tweet','label']].explode('tweet')
    val_tweets_labels = val_tweets_labels[~val_tweets_labels['tweet'].isnull()]

    if(args.subset_data):
        val_tweets_labels = val_tweets_labels[:10]

    if args.model == "gpt2":
        val_tweets = [tokenizer(tweet, truncation=True, max_length=1024, return_tensors='pt')['input_ids'].squeeze().to(device) for tweet in val_tweets_labels['tweet'].values]
        val_labels = val_tweets_labels['label'].values
    else:
        raise Exception("Bert is not supported right now!")

    print(f"{datetime.datetime.now()}: Padding sequences")
    val_tweets = pad_sequence(val_tweets).to(device).transpose(0, 1)

    print(f'{datetime.datetime.now()}: Cloning tweet tensor')
    val_masks = val_tweets.detach().clone()
    print(f"{datetime.datetime.now()}: Creating validation attention mask")
    val_masks = val_masks.cpu().apply_(lambda word: float(word>0)).to(device).float()

    print(f"{datetime.datetime.now()}: Tensorifying validation dataset")
    val_tweets_tensor = val_tweets.long()
    val_masks_tensor = val_masks
    val_labels_tensor = torch.LongTensor(val_labels).to(device)

    print(f"\n{datetime.datetime.now()}: Making test dataset")
    test_tweets_labels = test[['tweet','label']].explode('tweet')
    test_tweets_labels = test_tweets_labels[~test_tweets_labels['tweet'].isnull()]

    if(args.subset_data):
        test_tweets_labels = test_tweets_labels[:10]

    if args.model == "gpt2":
        test_tweets = [tokenizer(tweet, truncation=True, max_length=1024, return_tensors='pt')['input_ids'].squeeze().to(device) for tweet in test_tweets_labels['tweet'].values]
        test_labels = test_tweets_labels['label'].values
    else:
        raise Exception("Bert is not supported right now!")

    print(f"{datetime.datetime.now()}: Padding sequences")
    test_tweets = pad_sequence(test_tweets).to(device).transpose(0, 1)

    print(f'{datetime.datetime.now()}: Cloning tweet tensor')
    test_masks = test_tweets.detach().clone()
    print(f"{datetime.datetime.now()}: Creating test attention mask")
    test_masks = test_masks.cpu().apply_(lambda word: float(word>0)).to(device).float()

    print(f"{datetime.datetime.now()}: Tensorifying test dataset")
    test_tweets_tensor = test_tweets.long()
    test_masks_tensor = test_masks
    test_labels_tensor = torch.LongTensor(test_labels).to(device)

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

            tweets = tweets.unsqueeze(1)
            masks = masks.unsqueeze(1)

            loss = model(input_ids=tweets, token_type_ids=None, attention_mask=masks, mc_labels=labels)[0]
            
            avg_training_loss += loss.item()
            batches += 1

            loss.backward()
            optimizer.step()
        
        avg_training_loss = avg_training_loss / batches
        print(f'Average training loss: {avg_training_loss}\n')
        
        print("Validating model...")
        with torch.no_grad():
            model.eval()
            logits = model(val_tweets_tensor.unsqueeze(1), token_type_ids=None, attention_mask=val_masks_tensor)

            logits = logits[0].detach().cpu().numpy()
            label_ids = val_labels_tensor.to('cpu').numpy().flatten()

            validation_accuracy = np.sum(np.argmax(logits, axis=1).flatten() == label_ids) / len(label_ids)
            print(f'Validation accuracy: {validation_accuracy}\n')
        
    print("Saving model...")
    torch.save(model, f'{args.model}/{args.model}_BOT_DETECTION_{time.time()}.pt')

    # Evaluate the model.
    with torch.no_grad():
        model.eval()
        print('\nTesting model...')

        logits = model(test_tweets_tensor.unsqueeze(1), token_type_ids=None, attention_mask = test_masks_tensor)[0].detach().cpu().numpy()

        predictions = np.argmax(logits, axis=1)
        
        print(f'Summary statistics for gpt2 bot detection network.')
        print(classification_report(predictions, test_labels_tensor.cpu().numpy(), target_names=['human', 'bot']))

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for gradient descent.')
    parser.add_argument('--subset_data', type=bool, default=False, help="Whether to use a subset of the total data (smaller by factor of 10) for faster training.")
    parser.add_argument('--model', default='gpt2', choices=['bert', 'gpt2'])

    args = parser.parse_args()
    main(args)
