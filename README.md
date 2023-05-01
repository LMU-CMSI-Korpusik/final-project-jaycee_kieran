# Using Transformers To Detect Twitter Bots
By Kieran Ahn and Jaycee Nakagawa

This project uses two powerful language models, GPT-2 and BERT, to train classifiers that will classify tweets into human-or bot-generated.

In order to run this code, you must first acquire the TwiBot-20 dataset. For instructions on how to do so, please consult [this Github repository](https://github.com/BunsenFeng/TwiBot-20). Next, set up a virtual environment and run `pip3 install -r requirements.txt` to install all of the dependencies.

To train the models, use `python3 train.py`. GPT-2 or BERT can be selected with `--model gpt-2` or `--model bert` (default gpt-2). Learning rate and batch size can be set with `--learning_rate` and `--batch_size`, respectively. The number of users in the data can be reduced by a factor of 10 with `--subset_data True`. 

After the training loop is complete, the models will be saved to their individual folders, and a classification report will print to the console, and also be saved to the same directory. 

NOTE: as of the time of writing this README (5/1/2023), there exists a bug in the huggingface transformers library that breaks this code. To fix it, go to the GPT2DoubleHeadsModel in the `modeling_gpt2.py` in your installed transformers library and REMOVE the line which sets `num_labels = 1`.
