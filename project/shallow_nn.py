############################################################
#                                                          #
#                SHALLOW NEURAL NETWORK                    #
#                                                          #
############################################################

'''
DATA explained:
length of each key is 11250
'data': A list where each entry is an audio segment
'labels': A list where each entry is a 0-based integer label
'track_id': A list where each entry is a unique track id and all the audio segments belonging to the same track 
have the same id, useful for computing the maximum probability and  majority vote metrics.
track id example: [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1 etc]
theres 15 audio segments per track, all have the label per track
each audio segment has length 20462
'''


import sys
import numpy as np
import os
import os.path
import random
import tensorflow as tf
import pickle
from utils import melspectrogram

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', os.getcwd() + '/dataset/',
                            'Directory where the dataset will be stored and checkpoint. (default: %(default)s)')
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer('save_model', 1000,
                            'Number of steps between model saves (default: %(default)d)')

# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('batch_size', 256, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_float('learning_rate', 5e-05, 'Learning rate (default: %(default)d)')
tf.app.flags.DEFINE_integer('img_width', 32, 'Image width (default: %(default)d)')
tf.app.flags.DEFINE_integer('img_height', 32, 'Image height (default: %(default)d)')
tf.app.flags.DEFINE_integer('img_channels', 3, 'Image channels (default: %(default)d)')
tf.app.flags.DEFINE_integer('num_classes', 10, 'Number of classes (default: %(default)d)')
tf.app.flags.DEFINE_string('log_dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')


#run_log_dir = os.path.join(FLAGS.log_dir,
#                           'exp_bs_{bs}_lr_{lr}'.format(bs=FLAGS.batch_size,
#                                                        lr=FLAGS.learning_rate))
run_log_dir = os.path.join(FLAGS.log_dir, 'exp_DA1_bs_{bs}_lr_{lr}'.format(bs=FLAGS.batch_size, lr=FLAGS.learning_rate))


def parse_function(sounds, labels):
    sounds = melspectrogram(sounds)
    return sounds, labels

def batch_this(sounds,labels,batch_size):

    n_sounds = len(sounds)
    labels = tf.constant(labels)
    sounds = tf.constant(sounds)
    dataset = tf.data.Dataset.from_tensor_slices((sounds,labels))
    dataset = dataset.shuffle(buffer_size=n_sounds) 
    #dataset = dataset.map(parse_function)
    dataset = dataset.batch(batch_size).repeat()
    iterator = dataset.make_one_shot_iterator()
    return iterator




def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weights')

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='biases')

xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True)
def shallownn(x,is_training):
    
    #if is_training == 1:
     #   x_image = tf.map_fn(tf.image.random_flip_left_right,x_image)
    #img_summary = tf.summary.image('Input_images', x_image)
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=16,
        kernel_size=[10,23],
        padding='same',
        use_bias=False,
        #kernel_initializer=xavier_initializer,
        name='conv1'
    )
    #conv1_bn = tf.nn.relu(tf.layers.batch_normalization(conv1,training=is_training))
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[1, 20],
        strides=1,
        name='pool1'
    )
    conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding='same',
    use_bias=False,
    kernel_initializer=xavier_initializer,
    name='conv2'
    )
    conv2_bn = tf.nn.relu(tf.layers.batch_normalization(conv2,training=is_training))
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2_bn,
        pool_size=[2, 2],
        strides=2,
        name='pool2'
    )
    #print(np.shape(pool2))
    pool2_flat = tf.reshape(pool2, [-1,4096])
    fc1 = tf.layers.dense(pool2_flat,units=1024,activation=tf.nn.relu) 
    #print(np.shape(fc1))
    fcy = tf.layers.dense(fc1,units=10) 
    #h_final = tf.reshape(pool1, [-1,4096])

    return fcy

###############

#def main(_):
def main(_):
    tf.reset_default_graph()

    # Import data
    train_sounds = np.load('data/train_data_postmel.npy')
    train_labels = np.load('data/train_labels.npy')
    test_sounds = np.load('data/test_data_postmel.npy')
    test_labels = np.load('data/test_labels.npy')
    print('data Loaded')
    #Train split
    #train_sounds_pre = train_set['data']
    #train_sounds = []
    #for i in range(np.shape(train_sounds_pre)[0]):
    #for i in range(2):
     #   train_sounds.append(melspectrogram(train_sounds_pre[i][:]))
    #train_labels = test_set['labels']
    np.random.seed(0)
    np.random.shuffle(train_sounds)
    np.random.seed(0)
    np.random.shuffle(train_labels)
    #need np.asarray to remove all the ndarray types inside the array
    #a=np.asarray(train_sounds_pre)
    #train_sounds = []
    #print('before')
    #for i in range(np.shape(a)[0]):
    #   train_sounds.append(melspectrogram(a[i][:]))
    #   print(i)
    #print('here1')
    train_data = batch_this(train_sounds,train_labels,FLAGS.batch_size)
    train_batch = train_data.get_next()
    
    #Test split
    #test_sounds = test_set['data']
    #test_labels = test_set['labels']
    #np.random.seed(0)
    #np.random.shuffle(test_sounds)
    #np.random.seed(0)
    #np.random.shuffle(test_labels)
    test_data = batch_this(test_sounds,test_labels,FLAGS.batch_size)
    test_batch = test_data.get_next()

    
    with tf.variable_scope('inputs'):
        # Create the model
        x = tf.placeholder(tf.float32, [None, FLAGS.img_width * FLAGS.img_height * FLAGS.img_channels])
        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
    is_training = tf.placeholder(tf.bool)
    # Build the graph for the deep net
    y_conv = shallownn(x,is_training)

    with tf.variable_scope('x_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    
    # Define your AdamOptimiser, using FLAGS.learning_rate to minimixe the loss function
    #optimiser = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
    #global_step = tf.Variable(0, trainable=False)
    #learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,global_step ,1000,0.8)
    optimiser = tf.train.AdamOptimizer(FLAGS.learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08).minimize(cross_entropy)
    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #with tf.control_dependencies(update_ops):   
    #   optimiser = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
    # calculate the prediction and the accuracy
    #correct_prediction = tf.placeholder(tf.float32, [1])
    #accuracy = tf.Variable(tf.float32, [1])
    accuracy = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

    loss_summary = tf.summary.scalar('Loss', cross_entropy)
    acc_summary = tf.summary.scalar('Accuracy', accuracy)
    # summaries for TensorBoard visualisation
    validation_summary = tf.summary.merge([img_summary, acc_summary])
    training_summary = tf.summary.merge([img_summary, loss_summary])
    test_summary = tf.summary.merge([img_summary, acc_summary])
    # saver for checkpoints
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(run_log_dir + '_train', sess.graph,flush_secs=5)
        summary_writer_validation = tf.summary.FileWriter(run_log_dir +'_validate', sess.graph,flush_secs=5)
        
        sess.run(tf.global_variables_initializer())

        
        # Training and validation
        
        for step in range(1):
            # Training: Backpropagation using train set
            [train_sounds,train_labels] = sess.run(train_batch)
            train_labels = np.transpose(np.array([train_labels])) # makes it a column vector, required
            #validation
            #[validation_images,validation_labels] = sess.run(validation_batch)
            #validation_labels = np.transpose(np.array([validation_labels])) # makes it a column vector, required
            #test

            sess.run([optimiser], feed_dict={x: train_sounds, y_: train_labels})    
            
        '''
            _, summary_str = sess.run([optimiser, training_summary], feed_dict={x: trainImages, y_: trainLabels, is_training: True})
            
           
            if step % (FLAGS.log_frequency + 1)== 0:
                summary_writer.add_summary(summary_str, step)

            # Validation: Monitoring accuracy using validation set
            if step % FLAGS.log_frequency == 0:
                validation_accuracy, summary_str = sess.run([ accuracy,validation_summary], feed_dict={x: testImages, y_: testLabels, is_training: False})

                print('step %d, accuracy on validation batch: %g' % (step, validation_accuracy),sess.run(learning_rate))
                summary_writer_validation.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % FLAGS.save_model == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(run_log_dir + '_train', 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        # Testing

        # resetting the internal batch indexes
        cifar.reset()
        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0

        # don't loop back when we reach the end of the test set
        while evaluated_images != cifar.nTestSamples:
            (testImages, testLabels) = cifar.getTestBatch(allowSmallerBatches=True)
            test_accuracy_temp, _ = sess.run([accuracy, test_summary], feed_dict={x: testImages, y_: testLabels, is_training: False})

            batch_count = batch_count + 1
            test_accuracy = test_accuracy + test_accuracy_temp
            evaluated_images = evaluated_images + testLabels.shape[0]

        test_accuracy = test_accuracy / batch_count
        print('test set: accuracy on test set: %0.3f' % test_accuracy)
        '''
    print('done')



if __name__ == '__main__':
    tf.app.run(main=main)
