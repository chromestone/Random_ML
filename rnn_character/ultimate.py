# -*- coding: utf-8 -*-
"""ultimate training
@author Derek Zhang
"""
import tensorflow as tf
import numpy as np
import tensorflow.keras.utils
import tensorflow.keras.layers as KL
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as KC

VERBOSE = 2

# minus 1
EPOCH_START = 17
EPOCH_END = 256

BATCH_SIZE = 32
VAL_SIZE = 1025


with open("shakespear.txt") as f:
  text = f.read()

uniques = list(set(text))
# consistency for id generation
uniques.sort()

char2id = {k:i for i, k in enumerate(uniques)}

id2char = {i:k for i, k in enumerate(uniques)}

def to_tensor(char2id, text):
  
  tensor = np.zeros((len(text), len(char2id)))

  for i, e in enumerate(text):
    
    tensor[i, char2id[e]] = 1

  return tensor

def to_text(id2char, tensor):
  
  char_list = []
  assert len(id2char) == tensor.shape[1]
  
  first, second = np.where(tensor == 1)
  assert first.shape[0] == tensor.shape[0]

  for i in range(second.shape[0]):
    char_list.append(id2char[second[i]])
    
  return ''.join(char_list)

class DataGenerator(tensorflow.keras.utils.Sequence):
  
  ARBITRARY_LENGTH = 32

  'Generates data for Keras'
  def __init__(self, char2id, text, batch_size):
    'Initialization'
    self.char2id = char2id
    self.text = text
    self.batch_size = batch_size
    
    self.l = len(self.text) - DataGenerator.ARBITRARY_LENGTH - 1
    self.picker = np.arange(0, self.l)
    self.on_epoch_end()

    self.lenchar2id = len(char2id)

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(self.l / self.batch_size))

  def __getitem__(self, batch_index):
    'Generate one batch of data'

    start_index = batch_index * self.batch_size

    input_arr = np.zeros((self.batch_size,
                          DataGenerator.ARBITRARY_LENGTH, self.lenchar2id))
    target_arr = np.zeros((self.batch_size, self.lenchar2id))

    for j in range(self.batch_size):
      
      index = self.picker[start_index + j]

      for k, e in enumerate(self.text[index : index + DataGenerator.ARBITRARY_LENGTH]):

        input_arr[j, k, self.char2id[e]] = 1

      target_arr[j, self.char2id[self.text[index + DataGenerator.ARBITRARY_LENGTH]]] = 1

    return input_arr, target_arr

  def on_epoch_end(self):
      'Updates indexes after each epoch'

      np.random.shuffle(self.picker)

train_gen = DataGenerator(char2id, text[:-VAL_SIZE], batch_size = BATCH_SIZE)
val_gen = DataGenerator(char2id, text[-VAL_SIZE:], batch_size = BATCH_SIZE)

input_layer = KL.Input((DataGenerator.ARBITRARY_LENGTH, len(char2id)), name="the_input")

x = KL.LSTM(62, return_sequences=True, name="intermediate")(input_layer)
# used to do softmax but experimenting here
x = KL.LSTM(62, activation="relu", name="the_output")(x)

model = Model(input_layer, x)

model.summary()

def customLoss(yTrue,yPred):
  return tf.nn.softmax_cross_entropy_with_logits_v2(yTrue, yPred)
model.compile(loss=customLoss, optimizer='adam', metrics=['accuracy'])


model.load_weights('logs/weights.17-2.30.hdf5')

checkpt = KC.ModelCheckpoint('./logs/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)

history = model.fit_generator(train_gen, epochs=EPOCH_END, verbose=VERBOSE, validation_data=val_gen,
                    use_multiprocessing=True, initial_epoch=EPOCH_START,
                    callbacks=[checkpt], workers = 32, max_queue_size = 4096) #,
                    #class_weight = c_weights)


#from sklearn.utils import class_weight
#class_weights = class_weight.compute_class_weight('balanced',
#                                                 uniques,
#                                                 list(text))
#class_weights = class_weights * 1000
#c_weights = {i:class_weights[i] for i in range(class_weights.shape[0])}
