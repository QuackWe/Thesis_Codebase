import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Flatten, Concatenate, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from utils import BagDataGenerator, read_data
import helpers

# Set seed for reproducibility
tf.random.set_seed(42)

# Parameters
batch_size = 128
output_dim = 50
epochs = 10
learning_rate = 0.001

data_directory = "./data/"

# Load Data
data, data_train, data_valid, data_test = read_data(data_directory)

# Feature Dictionary (Removed time features)
feat_dic = {
    'cat_feat': ['cat_column1', 'cat_column2'],
    'num_feat': ['num_column1', 'num_column2']
}

# Generate Helpers
helpers_dic = helpers.get_helpers(data, feat_dic)

# Convert numerical features without splitting, removing square brackets
for col in feat_dic['num_feat']:
    data_train.loc[:, col] = data_train[col].apply(
        lambda x: [float(i.strip('[]')) for i in x.split(', ')] if isinstance(x, str) else x
    )
    data_valid.loc[:, col] = data_valid[col].apply(
        lambda x: [float(i.strip('[]')) for i in x.split(', ')] if isinstance(x, str) else x
    )
    data_test.loc[:, col] = data_test[col].apply(
        lambda x: [float(i.strip('[]')) for i in x.split(', ')] if isinstance(x, str) else x
    )

# Initialize Data Generators
train_generator = BagDataGenerator(data_frame=data_train,
                                   output_dim=output_dim,
                                   feat_dic=feat_dic,
                                   helpers_dic=helpers_dic,
                                   batch_size=batch_size,
                                   shuffle=True)

valid_generator = BagDataGenerator(data_frame=data_valid,
                                   output_dim=output_dim,
                                   feat_dic=feat_dic,
                                   helpers_dic=helpers_dic,
                                   batch_size=batch_size,
                                   shuffle=False)

# Model Definition
input_trace = Input(shape=(output_dim, helpers_dic['trace_helper']['vocab_size']), name='trace_input')
input_cat = Input(shape=(output_dim, len(feat_dic['cat_feat'])), name='cat_input')
input_num = Input(shape=(output_dim, len(feat_dic['num_feat'])), name='num_input')

# Concatenate inputs along the feature dimension
inputs_concat = Concatenate(axis=-1)([input_trace, input_cat, input_num])  # Shape: (batch_size, output_dim, total_feature_dim)

# LSTM layer with return_sequences=True
lstm_out = LSTM(64, return_sequences=True)(inputs_concat)

# TimeDistributed Dense layers
dense_out = TimeDistributed(Dense(32, activation='relu'))(lstm_out)
dropout_out = TimeDistributed(Dropout(0.5))(dense_out)

# Output layer with TimeDistributed
trace_output = TimeDistributed(Dense(helpers_dic['trace_helper']['vocab_size'], activation='softmax'), name='trace_out')(dropout_out)

# Compile Model
model = Model(inputs=[input_trace, input_cat, input_num], outputs=[trace_output])
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint(filepath='model_checkpoint.h5', save_best_only=True, monitor='val_loss', mode='min')

# Model Training
model.fit(train_generator,
          validation_data=valid_generator,
          epochs=epochs,
          callbacks=[checkpoint],
          verbose=1)

# Model Evaluation
test_generator = BagDataGenerator(data_frame=data_test,
                                  output_dim=output_dim,
                                  feat_dic=feat_dic,
                                  helpers_dic=helpers_dic,
                                  batch_size=batch_size,
                                  shuffle=False)

loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss}")
print(f"Test Trace Accuracy: {accuracy}")
