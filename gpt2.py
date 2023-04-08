"""
Author: Kieran Ahn

Follows the procedure presented in https://github.com/botonobot/Understanding-Transformers-for-Bot-Detection-Twitter/blob/master/bot_detection/bot_detection-BERT_GPT2-transformers.ipynb
"""

import datetime
import argparse
import numpy as np
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
import pandas as pd
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.metrics import classification_report

GPT_HIDDEN_DIM = 768
NUM_CLASSES = 2
BATCH_SIZE = 8
EPOCHS = 2
np.random.seed(1)
torch.manual_seed = (1)
torch.cuda.manual_seed_all(1)

class BotDetectionNet(nn.Module):
    def __init__(self, hidden_dim, num_y):
        super().__init__()
        self.gpt_cap = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, num_y),
            nn.Softmax()
        )

    def forward(self, gpt_weights):
        return self.gpt_cap(gpt_weights)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build the model.
    detector = BotDetectionNet(GPT_HIDDEN_DIM, NUM_CLASSES).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = AdamW(detector.parameters(), lr=args.learning_rate)

    print(f"{datetime.datetime.now()}: Initializing GPT-2")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True)
    gpt2 = GPT2DoubleHeadsModel.from_pretrained('gpt2', num_labels=NUM_CLASSES)
    gpt2.to(device)

    # Load the data.
    print(f"{datetime.datetime.now()}: Loading training data")
    train = pd.read_json("./Twibot-20/train.json")
    print(f"{datetime.datetime.now()}: Loading test data")
    test = pd.read_json("./Twibot-20/test.json")
    print(f"{datetime.datetime.now()}: Loading validaton data")
    validation = pd.read_json("./Twibot-20/dev.json")

    print(f'\n{datetime.datetime.now()}: Making training dataset')
    train_tweets_labels = train[['tweet', 'label']].explode('tweet')
    train_tweets_labels = train_tweets_labels[~train_tweets_labels['tweet'].isnull()]
    
    train_tweets = [tokenizer(tweet, truncation=True, max_length=1024, return_tensors='pt')['input_ids'].squeeze().to(device) for tweet in train_tweets_labels['tweet'].values]
    
    print(f'{datetime.datetime.now()}: Padding sequences')
    train_tweets = pad_sequence(train_tweets).to(device).transpose(0, 1)
    train_labels = train_tweets_labels['label'].values

    print(f'{datetime.datetime.now()}: Cloning tweet tensor')
    train_masks = train_tweets.detach().clone()
    print(f"{datetime.datetime.now()}: Creating training attention mask")
    train_masks.cpu().apply_(lambda word: float(word>0)).to(device).long().to(device)

    print(f"{datetime.datetime.now()}: Tensorifying training dataset")
    train_tweets.long()
    train_masks_tensor = train_masks
    train_labels_tensor = torch.LongTensor(train_labels).to(device)

    print(f"{datetime.datetime.now()}: Importing to TensorDataset")
    train_dataset = TensorDataset(train_tweets, train_masks_tensor, train_labels_tensor)

    print(f"{datetime.datetime.now()}: Making training dataloader")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    print(f"\n{datetime.datetime.now()}: Making validation dataset")
    val_tweets_labels = validation[['tweet','label']].explode('tweet')
    val_tweets_labels = val_tweets_labels[~val_tweets_labels['tweet'].isnull()]
    val_tweets = [tokenizer(tweet, truncation=True, max_length=1024, return_tensors='pt')['input_ids'].squeeze().to(device) for tweet in val_tweets_labels['tweet'].values]

    print(f"{datetime.datetime.now()}: Padding sequences")
    val_tweets = pad_sequence(val_tweets).to(device).transpose(0, 1)
    val_labels = val_tweets_labels['label'].values

    print(f'{datetime.datetime.now()}: Cloning tweet tensor')
    val_masks = val_tweets.detach().clone()
    print(f"{datetime.datetime.now()}: Creating validation attention mask")
    val_masks.cpu().apply_(lambda word: float(word>0)).to(device).long()

    print(f"{datetime.datetime.now()}: Tensorifying validation dataset")
    val_tweets.long()
    val_masks_tensor = val_masks
    val_labels_tensor = torch.LongTensor(val_labels).to(device)

    print(f"\n{datetime.datetime.now()}: Making test dataset")
    test_tweets_labels = test[['tweet','label']].explode('tweet')
    test_tweets_labels = test_tweets_labels[~test_tweets_labels['tweet'].isnull()]

    test_tweets = [tokenizer(tweet, truncation=True, max_length=1024, return_tensors='pt')['input_ids'].squeeze().to(device) for tweet in test_tweets_labels['tweet'].values]

    print(f"{datetime.datetime.now()}: Padding sequences")
    test_tweets = pad_sequence(test_tweets).to(device).transpose(0, 1)
    test_labels = test_tweets_labels['label'].values

    print(f'{datetime.datetime.now()}: Cloning tweet tensor')
    test_masks = test_tweets.detach().clone()
    print(f"{datetime.datetime.now()}: Creating test attention mask")
    test_masks.cpu().apply_(lambda word: float(word>0)).to(device).long()

    print(f"{datetime.datetime.now()}: Tensorifying test dataset")
    test_tweets.long()
    test_masks_tensor = test_masks
    test_labels_tensor = torch.LongTensor(test_labels).to(device)

    # Train the model for the specified number of epochs.
    for epoch in range(EPOCHS):
        gpt2.train()
        print(f'Epoch {epoch+1}')
        print('Training model...')
        train_iterator = iter(train_dataloader)
        avg_training_loss = 0

        batches = 0
        for tweets, masks, labels in tqdm(train_iterator):
            tweets = tweets.reshape(8, 1, 783)
            masks = masks.reshape(8, 1, 783)
            labels = labels.unsqueeze(0).reshape(8, 1)
            loss = gpt2(tweets, token_type_ids=None, attention_mask=masks, mc_labels=labels)
            avg_training_loss += loss[0].item()
            batches += 1

            loss[0].backward()
            optimizer.step()
            optimizer.zero_grad()
        
        avg_training_loss = avg_training_loss / batches
        print(f'Average training loss: {avg_training_loss}\n')

        
        print("Validating model...")
        with torch.no_grad():
            logits = gpt2(val_tweets.reshape(len(val_tweets), 1, 783), token_type_ids=None, attention_mask=val_masks_tensor.reshape(len(val_masks_tensor), 1, 783))

            logits = logits[0].detach().cpu().numpy()
            label_ids = val_labels_tensor.to('cpu').numpy().flatten()

            validation_accuracy = np.sum(np.argmax(logits, axis=1).flatten() == label_ids) / len(label_ids)
            print(f'Validation accuracy: {validation_accuracy}\n')
        
    print("Saving model...")
    torch.save(gpt2, f'./GPT-2/GPT_2_BOT_DETECTION_{datetime.datetime.now()}')

    # Evaluate the model.
    with torch.no_grad():
        gpt2.eval()
        print('\nTesting model...')

        logits = gpt2(test_tweets.reshape(len(test_tweets), 1, 783), token_type_ids=None, attention_mask = test_masks_tensor.reshape(len(test_masks_tensor), 1, 783))[0].detach().cpu().numpy()

        predictions = np.argmax(logits, axis=1)
        
        print(f'Summary statistics for gpt2 bot detection network.')
        print(classification_report(predictions, test_labels_tensor.numpy(), target_names=['human', 'bot']))

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for gradient descent.')

    args = parser.parse_args()
    main(args)
