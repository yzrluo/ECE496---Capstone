import os
# set following var to "0" for GPU and ""(empty string) for CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import csv
import numpy
import pandas as pd
from itertools import islice

myfile = "./data/training_samples_2k.csv"
myfile_labels = "./data/ouput_samples_2k.csv"

test_inputs_file = "./data/training_samples_002k.csv"
test_output_file = "./data/ouput_samples_002k.csv"

# Mini-batch GD Parameters
learning_rate = 0.0000003#0.00003
major_steps = 300
num_steps = 200
batch_size = 16
mini_batch_size = 64
display_step = 100
scaling = 0.001

# Network Parameters
n_hidden_1 = 1000 # 1st layer number of neurons
n_hidden_2 = 64 # 2nd layer number of neurons
n_hidden_3 = 256 # 2nd layer number of neurons
num_input = 262144 # Time/Spatial-domain signal
num_classes = 262144 # Spectrum

with open(test_inputs_file, "r") as f_input:
    reader_input = csv.reader(f_input, delimiter=",")
    x_input = list(reader_input)
    time_domain_test = numpy.array(x_input).astype("float")
    
time_domain_test=time_domain_test*scaling
    
with open(test_output_file, "r") as f:
    reader = csv.reader(f, delimiter=",")
    x = list(reader)
    spectrum_test = numpy.array(x).astype("float")
    
spectrum_test=spectrum_test*scaling
    
# Define the neural network
def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['images']
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)    
    # Hidden fully connected layer with 256 neurons
    layer_3 = tf.layers.dense(layer_2, n_hidden_3)
    
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_1, num_classes)
    return out_layer
    
def model_fn(features, labels, mode):
    # Build the neural network
    predicted_s = neural_net(features)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predicted_s )

    # Define losstensorflow.python.time_domain_testtraining.basic_session_run_hooks.NanLossDuringTrainingError: NaN loss during training.
    loss_op = tf.reduce_mean(tf.squared_difference(x=tf.cast(labels, dtype=tf.float32), y=tf.cast(predicted_s , dtype=tf.float32)));

    # Define optimizer
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    '''
    optimizer = tf.train.AdagradOptimizer(
        learning_rate=0.0001,
        initial_accumulator_value=0.1,
        use_locking=False,
        name='Adagrad'
    )
    '''
    '''
    optimizer = tf.train.MomentumOptimizer(
            learning_rate=0.00003,
            momentum=0.2,
            use_locking=False,
            name='Momentum',
            use_nesterov=False
    )
    '''
    
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate=1,
        rho=0.90,
        epsilon=1e-08,
        use_locking=False,
        name='Adadelta'
    )
    
    '''
    optimizer = tf.train.AdamOptimizer(
        learning_rate=0.08,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        use_locking=False,
        name='Adam'
    )
    '''


    
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=predicted_s )

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predicted_s,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Prepare matrices for first-time training
start = 0
fd = open(myfile)
reader = csv.reader(fd,delimiter=",")
batch_rows = [row for idx, row in enumerate(reader) if ((idx >= start) 
    and (idx < start + mini_batch_size))]
x_input = list(batch_rows)
spatial_domain_mini = numpy.array(x_input).astype("float")
spatial_domain_mini = spatial_domain_mini*scaling
fd.close()

fd = open(myfile_labels)
reader = csv.reader(fd,delimiter=",")
batch_rows = [row for idx, row in enumerate(reader) if ((idx >= start) 
    and (idx < start + mini_batch_size))]
x_input = list(batch_rows)
spectrum_mini = numpy.array(x_input).astype("float")
spectrum_mini = spectrum_mini*scaling
fd.close()

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': spatial_domain_mini}, y=spectrum_mini,
    batch_size=batch_size, num_epochs=None, shuffle=True)
    
# Train the Model
model.train(input_fn, steps=num_steps)


# Mini-batch
for i in range(1,major_steps):
    print("major step: ",i)
    
    start = i*mini_batch_size
    fd = open(myfile)
    reader = csv.reader(fd,delimiter=",")
    batch_rows = [row for idx, row in enumerate(reader) if ((idx >= start) 
        and (idx < start + mini_batch_size))]
    x_input = list(batch_rows)
    spatial_domain_mini = numpy.array(x_input).astype("float")
    spatial_domain_mini = spatial_domain_mini*scaling
    fd.close()

    fd = open(myfile_labels)
    reader = csv.reader(fd,delimiter=",")
    batch_rows = [row for idx, row in enumerate(reader) if ((idx >= start) 
        and (idx < start + mini_batch_size))]
    x_input = list(batch_rows)
    spectrum_mini = numpy.array(x_input).astype("float")
    spectrum_mini = spectrum_mini*scaling
    fd.close()

    input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': spatial_domain_mini}, y=spectrum_mini,
    batch_size=batch_size, num_epochs=None, shuffle=True)
    
    new_model = model
    model.train(input_fn, steps=num_steps)
    model = new_model



# Predict
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"images": time_domain_test},
    num_epochs=1,
    shuffle=False)

predictions = list(model.predict(input_fn=predict_input_fn))

# extract weights
para_names = model.get_variable_names()
input_layer_1_bias = model.get_variable_value('dense/bias')
input_layer_1_kernal = (model.get_variable_value('dense/kernel')).T
layer_1_2_bias = model.get_variable_value('dense_1/bias')
layer_1_2_kernal = (model.get_variable_value('dense_1/kernel')).T
#layer_2_3_bias = model.get_variable_value('dense_2/bias')
#layer_2_3_kernal = (model.get_variable_value('dense_2/kernel')).T

# write weights into files
f = open('./data/weights/input_layer_1_bias.txt','w')
for i in range(0,n_hidden_1):
    f.write(str(input_layer_1_bias[i])+'\n')
f.close()

f = open('./data/weights/input_layer_1_kernal.txt','w')
for i in range(0,n_hidden_1):
        for j in range(0,num_input):
            f.write(str(input_layer_1_kernal[i][j])+'\n')
f.close()

f = open('./data/weights/layer_1_2_bias.txt','w')
for i in range(0,n_hidden_2):
    f.write(str(layer_1_2_bias[i])+'\n')
f.close()

f = open('./data/weights/layer_1_2_kernal.txt','w')
for i in range(0,n_hidden_2):
        for j in range(0,n_hidden_1):
            f.write(str(layer_1_2_kernal[i][j])+'\n')
f.close()
'''
f = open('./data/weights/layer_2_3_bias.txt','w')
for i in range(0,num_classes):
    f.write(str(layer_2_3_bias[i])+'\n')
f.close()

f = open('./data/weights/layer_2_3_kernal.txt','w')
for i in range(0,num_classes):
        for j in range(0,n_hidden_1):
            f.write(str(layer_2_3_kernal[i][j])+'\n')
'''
f.close()
