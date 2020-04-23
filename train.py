from preprocess import *
from sklearn.model_selection import train_test_split

training_comments_file = "data/train.from"
training_replies_file = "data/train.to"
model_dir = "data/seq2seq_model"

# load the training comments and replies and clean them
comments = collect_comments(training_comments_file)
replies = collect_comments(training_replies_file)

cleaned_comments = clean_sentences(comments)
cleaned_replies = clean_sentences(replies)

# fit the tokenizer to all the sentences in the training set
vocab_size = fit_tokenizer(cleaned_comments, cleaned_replies)

# tokenize the input and output sentences
input_tensor = tokenize(cleaned_comments)
target_tensor = tokenize(cleaned_replies)

input_length = input_tensor.shape[1]
target_length = target_tensor.shape[1]

# split into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(input_tensor, target_tensor, test_size=0.20)

print("Training Examples: {}".format(len(x_train)))
print("Validation Examples: {}".format(len(x_val)))

buffer_size = 1000
batch_size = 64
steps_per_epoch = len(x_train) // batch_size
embedding_dim = 256
units = 1024

