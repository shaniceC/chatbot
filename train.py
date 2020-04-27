import os
import time
from preprocess import *
from seq2seq import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy

training_comments_file = "data/train.from"
training_replies_file = "data/train.to"
model_dir = "data/seq2seq_model.h5"

# load the training comments and replies and clean them
print("Loading training comments and replies")
comments = collect_comments(training_comments_file)
replies = collect_comments(training_replies_file)

cleaned_comments = clean_sentences(comments)
cleaned_replies = clean_sentences(replies)

# fit the tokenizer to all the sentences in the training set
print("Fitting the tokenizer")
vocab_size = fit_tokenizer(cleaned_comments, cleaned_replies)

# tokenize the input and output sentences
print("Tokenizing the sentences")
input_tensor = tokenize(cleaned_comments)
target_tensor = tokenize(cleaned_replies)

input_length = input_tensor.shape[1]
target_length = target_tensor.shape[1]

# split into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(input_tensor, target_tensor, test_size=0.20)

print("Training Examples: {}".format(len(x_train)))
print("Validation Examples: {}".format(len(x_val)))

buffer_size = 500
batch_size = 64
steps_per_epoch = len(x_train) // batch_size
embedding_dim = 256
units = 256
epochs = 10

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size)
dataset = dataset.batch(batch_size, drop_remainder=True)

optimizer = Adam()
encoder = Encoder(vocab_size, embedding_dim, units, batch_size)
attention_layer = BahdanauAttention(10)
decoder = Decoder(vocab_size, embedding_dim, units, batch_size)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoin(optimizer=optimizer, encoder=encoder, decoder=decoder)

# train the model
print("Training the model")
for epoch in range(epochs):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inpt, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inpt, targ, enc_hidden, encoder, decoder, optimizer)
        total_loss += batch_loss

        if batch % 100 == 0:
            print("Epoch: {} Batch: {} Loss: {}".format(epoch + 1, batch, batch_loss.numpy()))


    # save checkpoint of the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print("Epoch: {} Loss: {:.4f}".format(epoch + 1, total_loss / steps_per_epoch))
    print("Time taken for 1 epoch: {} sec\n".format(time.time() - start))
