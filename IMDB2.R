library(keras)
library(tidyverse)
library(caret)
memory.limit(size=200000)
max_words <- 20000 


#Preparing data
imdb_dir <- "~/Downloads/aclImdb"
data_dir <- file.path(imdb_dir, "data")
labels <- c()
texts <- c()
# downloads reviews' text in texts while corresponding
#sentiments are stored in labels, 0=negative, 1= positive
for (label_type in c("neg", "pos")) {
  label <- switch(label_type, neg = 0, pos = 1)
  dir_name <- file.path(data_dir, label_type)
  for (fname in list.files(dir_name, pattern = glob2rx("*.txt"), 
                           full.names = TRUE)) {
    texts <- c(texts, readChar(fname, file.info(fname)$size))
    labels <- c(labels, label)
  }
}
# Now we tokenize the texts read from the files
# We restrict it to max 20000 words

tokenizer <- text_tokenizer(num_words = max_words) %>% 
  fit_text_tokenizer(texts)
# tokenize sequences (vectors of integers) are generated
# and stored in a list sequences
sequences <- texts_to_sequences(tokenizer, texts)
word_index = tokenizer$word_index
cat("Found", length(word_index), "unique tokens.\n")
#Found 115970 unique tokens.
# train_sequences will be used later on with other models
# where padding is required
x<-sequences
y<-as.numeric(labels)
# removes extra variables to clear memory
rm(texts,sequences,labels)
set.seed(100)
#since x is a list of integer vectors, we are using y
# to generate test_index
test_index<-createDataPartition(y,times=1,p=0.4,list=FALSE)
x_train<-x[-test_index]
y_train<-y[-test_index]
x_test<-x[test_index]
y_test<-y[test_index]

#Max no of words in any review
length(train_sequences[[which.max(lapply(train_sequences,length))]])
# Following creates a data frame of number of words in 25000 reviews
temp<-unlist(lapply(train_sequences,length),use.names = FALSE)
tempdf<-data.frame(nwords=temp)
# If we select max review length as 500, % of reviews would be truncated
tempdf%>%filter(nwords>500)%>%summarise(n=n()/length(train_sequences))
# If we select max review length as 800, % of reviews would be truncated
tempdf%>%filter(nwords>800)%>%summarise(n=n()/length(train_sequences))
################################################
# Preparing data for dense Feedforward models
################################################
# For FF model, we need to convert integer vectors representing
#review texts into a tensor of shape (samples, vector_sequence)
# where samples are no of rows of matrix and vector_sequence
# is the vector in sequences list converted into a one-hot
# encoding row of dimention=max_words. Only those columns will
#be 1 in a row (represnting a review) whose corresting word
# index is found in the row, all other columns will be 0

# First we define a function to convert list of integer vectors
# into a matrix (tensor)
vectorize_sequences <- function(sequences, dimension = max_words) {
  # Create an all-zero matrix of shape (len(sequences), dimension)
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    # Sets specific indices of results[i] to 1s
    results[i, sequences[[i]]] <- 1
  results
}
# Now convert x_train and y_train list of integer vectors
# into a tensor or shape (samples, vector_sequence)
x_train <- vectorize_sequences(x_train)
x_test<-vectorize_sequences(x_test)
# Simple Feedforward model
################################################
model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(max_words)) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")
summary(model)
# Compiling the model with rmsprop optimizer
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "binary_crossentropy",
  metrics = c("acc")
)
history <- model %>% fit(
  x_train, y_train,
  shuffle=TRUE,
  epochs = 20,
  batch_size = 512,
  validation_split = 0.2
)

epoch<-which.max(history$metrics$val_acc)
epoch
#3
max(history$metrics$val_acc)
#0.8394798
model %>% fit(x_train, y_train, epochs = epoch, batch_size = 512)
results <- model %>% evaluate(x_test, y_test)
results
#loss accuracy 
#0.8891655 0.8626754
###################################################
#Feedforward model with dropout
###################################################
k_clear_session()
model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(max_words )) %>% 
  layer_dropout(rate=0.5)%>%
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dropout(rate=0.5)%>%  
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
) 
history <- model %>% fit(
  x_train, y_train,
  shuffle=TRUE,
  epochs = 20,
  batch_size = 512,
  validation_split = 0.2
)
epoch<-which.max(history$metrics$val_accuracy)
epoch
#9
max(history$metrics$val_acc)
#0.8381211
model %>% fit(x_train, y_train, epochs = epoch, batch_size = 512)
results <- model %>% evaluate(x_test, y_test)
results
#loss  accuracy 
#0.5989639 0.8790402
###################################################
# Feedforward model with kernal L2 regularization
###################################################
k_clear_session()
model <-keras_model_sequential() %>%
  layer_dense(units = 16, kernel_regularizer = regularizer_l2(0.01),
  activation = "relu", input_shape = c(max_words)) %>%
  layer_dense(units = 16, kernel_regularizer = regularizer_l2(0.01),
  activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
) 
history <- model %>% fit(
  x_train, y_train,
  shuffle=TRUE,
  epochs = 20,
  batch_size = 512,
  validation_split = 0.2
)
epoch<-which.max(history$metrics$val_accuracy)
epoch
#16
max(history$metrics$val_acc)
#0.9318711
model %>% fit(x_train, y_train, epochs = epoch, batch_size = 512)
results <- model %>% evaluate(x_test, y_test)
results
#loss  accuracy 
#0.3880340 0.8787491  
###################################################
# Prepareing data for rest of the models
###################################################
# For rest of the models, we don't use one-hot encoding
# rather we use list of integer vectors in x_train and x_test
# each reprsenting a review and pad them with trailing 0's
# if they are short of max_len, if the no of words in review
# are more than max_len, we truncate rest of the words
x_train<-x[-test_index]
y_train<-y[-test_index]
x_test<-x[test_index]
y_test<-y[test_index]
max_features <- 20000  # Number of words to consider as features
maxlen <- 500          # Cut texts after this number of words 
# Pad sequences
x_train<-pad_sequences(x_train, maxlen = maxlen)
x_test<-pad_sequences(x_test, maxlen = maxlen)

####################################################
# with 2 bidirectional LSTMs
####################################################
k_clear_session()

model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 32) %>% 
  layer_dropout(0.5)%>%
  bidirectional(
    layer_lstm(units = 64, return_sequences=TRUE)
  ) %>% 
  bidirectional(
    layer_lstm(units = 64)
  ) %>% 
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "binary_crossentropy",
  metrics = c("acc")
)
history <- model %>% fit(
  x_train, y_train,
  epochs = 4,
  batch_size = 128,
  validation_split = 0.2
)
epoch<-which.max(history$metrics$val_acc)
epoch
#2
max(history$metrics$val_acc)
#0.9429348
model %>% fit(x_train, y_train, epochs = epoch, batch_size = 128)
results <- model %>% evaluate(x_test, y_test)
results
#loss       acc 
#0.2700873 0.8940656  
############################################
# 1D conv
###########################################
k_clear_session()

model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 128,
                  input_length = maxlen) %>% 
  layer_dropout(0.2)%>%
  layer_conv_1d(filters = 250, kernel_size = 3,
                activation = "relu") %>% 
  layer_max_pooling_1d(pool_size = 5) %>% 
  layer_conv_1d(filters = 250, kernel_size = 3, 
                activation = "relu") %>% 
  layer_global_max_pooling_1d() %>% 
  layer_dense(units = 1,activation = "sigmoid")
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "binary_crossentropy",
  metrics = c("acc")
)
history <- model %>% fit(
  x_train, y_train,
  epochs = 8,
  batch_size = 32,
  validation_split = 0.2
)
epoch<-which.max(history$metrics$val_acc)
epoch
#6
max(history$metrics$val_acc)
#0.9437112
model %>% fit(x_train, y_train, epochs = epoch, batch_size = 32)
results <- model %>% evaluate(x_test, y_test)
results
#loss      acc 
#0.7355296 0.8763613 
################################################
# 1D conv with 2 bidirectional LSTMs
################################################
k_clear_session()

model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 128,
                  input_length = maxlen) %>% 
  layer_dropout(0.5)%>%
  layer_conv_1d(filters = 250, kernel_size = 3, activation = "relu") %>% 
  layer_max_pooling_1d(pool_size = 5) %>% 
  layer_conv_1d(filters = 250, kernel_size = 3, activation = "relu") %>% 
  bidirectional(
    layer_lstm(units = 64, return_sequences=TRUE)
  ) %>% 
  bidirectional(
    layer_lstm(units = 64)
  ) %>% 
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "binary_crossentropy",
  metrics = c("acc")
)
history <- model %>% fit(
  x_train, y_train,
  epochs = 5,
  batch_size = 32,
  validation_split = 0.2
)
epoch<-which.max(history$metrics$val_acc)
epoch
#1
max(history$metrics$val_acc)
#0.9227484
model %>% fit(x_train, y_train, epochs = epoch, batch_size = 32)
results <- model %>% evaluate(x_test, y_test)
results
#loss  accuracy
#0.3024800 0.8924931
##############################################
# With Glove
##############################################
glove_dir = '~/Downloads/glove.6B'
lines <- readLines(file.path(glove_dir, "glove.6B.100d.txt"))
embeddings_index <- new.env(hash = TRUE, parent = emptyenv())
for (i in 1:length(lines)) {
  line <- lines[[i]]
  values <- strsplit(line, " ")[[1]]
  word <- values[[1]]
  embeddings_index[[word]] <- as.double(values[-1])
}
embedding_dim <- 100
embedding_matrix <- array(0, c(max_words, embedding_dim))
for (word in names(word_index)) {
  index <- word_index[[word]]
  if (index < max_words) {
    embedding_vector <- embeddings_index[[word]]
    if (!is.null(embedding_vector))
      # Words not found in the embedding index will be all zeros.
      embedding_matrix[index+1,] <- embedding_vector
  }
}
model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 100,
                  input_length = maxlen) %>% 
  layer_dropout(0.2)%>%
  layer_conv_1d(filters = 250, kernel_size = 3, activation = "relu") %>% 
  layer_max_pooling_1d(pool_size = 5) %>% 
  layer_conv_1d(filters = 250, kernel_size = 3, activation = "relu") %>% 
  bidirectional(
    layer_lstm(units = 64, return_sequences=TRUE)
  ) %>% 
  bidirectional(
    layer_lstm(units = 64)
  ) %>% 
  layer_dense(units = 1, activation = "sigmoid")
summary(model)
# Loading pretrained word embeddings into the embedding layer
get_layer(model, index = 1) %>%
  set_weights(list(embedding_matrix)) %>%
  freeze_weights()
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "binary_crossentropy",
  metrics = c("acc")
)
history <- model %>% fit(
  x_train, y_train,
  epochs = 5,
  batch_size = 32,
  validation_split = 0.2
)
epoch<-which.max(history$metrics$val_acc)
epoch
#1
max(history$metrics$val_acc)
#0.9227484
model %>% fit(x_train, y_train, epochs = epoch, batch_size = 32)
results <- model %>% evaluate(x_test, y_test)
results
#loss  accuracy
#0.4710467 0.8813698