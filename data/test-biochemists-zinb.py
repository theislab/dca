import numpy as np
from autoencoder.io import read_text
from autoencoder.network import mlp

count = read_text('biochemists.tsv', header='infer')
y = count[:, 0].astype(int)
x = count[:, 1:]

net = mlp(x.shape[1], output_size=1, hidden_size=(), masking=False, loss_type='zinb')
model = net['model']

model.summary()
model.compile(loss=net['loss'], optimizer='Adam')
model.fit(x, y, epochs=700, batch_size=32)

print('Theta: %f' % (1/np.exp(model.get_weights()[2][0][0])))
