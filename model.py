import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input
from tensorflow.keras.models import Model


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


inputs = Input(shape=(784,))
outputs = Dense(10, activation='softmax')(inputs)
model = Model(inputs=inputs, outputs=outputs)


model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
       loss='categorical_crossentropy',
       metrics=['accuracy'])

# Training the model
model.fit(x_train, y_train, 
    batch_size=100,
    epochs=2,
    validation_data=(x_test, y_test))

# Evaluation
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('Test accuracy:', test_accuracy)
