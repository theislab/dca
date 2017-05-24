import numpy as np
import tensorflow as tf
from autoencoder.io import read_text, preprocess
from autoencoder.api import autoencode
import keras.backend as K

# for full reproducibility
np.random.seed(1)
tf.set_random_seed(1)
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1,
                                        inter_op_parallelism_threads=1))
K.set_session(sess)

x = read_text('biochemists.tsv', header='infer')
print(x.shape)

# test API
result = autoencode(x, 'test-ae', type='zinb-conddisp', hidden_size=(1,), epochs=3)
