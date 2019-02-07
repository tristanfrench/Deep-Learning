import sys
import numpy as np
import os
import os.path
import random
import tensorflow as tf
import pickle
import load_data
from shallow_nn import shallow
from deep_nn import deep
np.set_printoptions(suppress=True)
np.set_printoptions(1)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_epochs', 10,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_string('model_type', 'shallow','Type of model used, shallow or deep. (default: %(default)s)')
tf.app.flags.DEFINE_string('learning_type', 'normal','Decaying learning rate or not. (default: %(default)s)')
tf.app.flags.DEFINE_string('initialiser', 'normal','Xavier initialiser or not. (default: %(default)s)')
tf.app.flags.DEFINE_string('batch_norm', 'False','batch norm or not. (default: %(default)s)')
tf.app.flags.DEFINE_integer('batch_size', 16, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_float('learning_rate', 5e-05, 'Learning rate (default: %(default)d)')
tf.app.flags.DEFINE_string('log_dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')
run_log_dir = os.path.join(FLAGS.log_dir, '{m}_bs_{bs}_{lt}'.format(m=FLAGS.model_type, bs=FLAGS.batch_size, lt=FLAGS.learning_type))




def batch_this(sounds, labels, batch_size, n_sounds, repeat=0, track_id=0, track_tag=0):
    '''
    Returns a batched iterator containing data and label
    '''
    #Create tensor from data
    if track_id==0:
        dataset = tf.data.Dataset.from_tensor_slices((sounds,labels))
    elif track_tag==0:
        dataset = tf.contrib.data.Dataset.from_tensor_slices((sounds,labels,track_id))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((sounds,labels,track_id,track_tag))
    #Shuffle data
    dataset = dataset.shuffle(buffer_size=n_sounds) 
    #Repeat data
    if repeat:
        dataset = dataset.batch(batch_size).repeat()
    else:
        dataset = dataset.batch(batch_size)
    #Make iterator
    iterator = dataset.make_initializable_iterator()
    return iterator


###############

def main(_):
    tf.reset_default_graph()
    # Import data
    train_data_sounds,test_data_sounds,train_data_labels,test_data_labels_og,test_data_id = load_data.main()
    print('Data Loaded')
    test_data_tag = np.arange(3750)
    img_number = len(test_data_labels_og)
    log_per_epoch = int(1000/FLAGS.max_epochs)
    log_frequency = len(train_data_labels)/FLAGS.batch_size/log_per_epoch
    if log_frequency == 0 :
        log_frequency+=1
    #Data placeholders
    train_data_placeholder = tf.placeholder(tf.float32, [None, 80, 80])
    train_labels_placeholder = tf.placeholder(tf.int32, [None])
    test_data_placeholder = tf.placeholder(tf.float32, [None, 80, 80])
    test_labels_placeholder = tf.placeholder(tf.int32, [None])
    test_id_placeholder = tf.placeholder(tf.int32, [None])
    test_tag_placeholder = tf.placeholder(tf.int32, [None])
    #Train split
    n_sounds = len(train_data_labels)
    train_iterator = batch_this(train_data_placeholder,train_labels_placeholder,FLAGS.batch_size,n_sounds)
    train_batch = train_iterator.get_next()
    #Test split
    test_data_labels = test_data_labels_og
    test_iterator = batch_this(test_data_placeholder,test_labels_placeholder,FLAGS.batch_size,len(test_data_labels),1)
    test_batch = test_iterator.get_next()
    print('Data preprocessing done')
    
    with tf.variable_scope('inputs'):
        x = tf.placeholder(tf.float32, [None,80,80])
        y_ = tf.placeholder(tf.int32, [None])

    #Deep or shallow model
    if FLAGS.model_type == 'shallow':
        y_conv = shallow(x,FLAGS.initialiser)
    elif FLAGS.model_type == 'deep':
        y_conv = deep(x,FLAGS.initialiser)
    #Loss
    with tf.variable_scope('x_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    #L1 regularization
    l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.0001)
    all_weights = tf.trainable_variables()
    regularization_factor = tf.contrib.layers.apply_regularization(l1_regularizer, weights_list= all_weights)
    cross_entropy += regularization_factor 
    #Learning rate + Adam optimizer
    if FLAGS.learning_type == 'decay':
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,global_step ,1000,0.8)
        optimiser = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)
    elif FLAGS.learning_type == 'normal':
        optimiser = tf.train.AdamOptimizer(FLAGS.learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08).minimize(cross_entropy)
    #Accuracy
    accuracy = tf.equal(tf.argmax(y_conv,1),tf.cast(y_, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
    #Summaries
    loss_summary = tf.summary.scalar('Loss', cross_entropy)
    acc_summary = tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge([loss_summary,acc_summary])
    
    with tf.Session() as sess:
        #Summary writers
        train_writer = tf.summary.FileWriter(run_log_dir + '_train', sess.graph,flush_secs=5)
        test_writer = tf.summary.FileWriter(run_log_dir + '_test', sess.graph,flush_secs=5)
        sess.run(tf.global_variables_initializer())
        step = 0
        #TRAINING LOOP
        for epoch in range(FLAGS.max_epochs):
            #Run initializers
            sess.run(train_iterator.initializer,feed_dict={train_data_placeholder:train_data_sounds, train_labels_placeholder:train_data_labels})
            sess.run(test_iterator.initializer,feed_dict={test_data_placeholder:test_data_sounds, test_labels_placeholder:test_data_labels})
            print('Epoch:',epoch)
            while True:
                #Train optimizer for all batches, stop when out of range
                try:
                    [train_sounds,train_labels] = sess.run(train_batch)
                    _,train_summary = sess.run([optimiser,merged], feed_dict={x:train_sounds, y_:train_labels}) 
                except tf.errors.OutOfRangeError:
                    break
                #Print Accuracy on test set
                [test_sounds,test_labels] = sess.run(test_batch)
                validation_accuracy,test_summary = sess.run([accuracy,merged], feed_dict={x:test_sounds, y_:test_labels})
                if step % 170 == 0:
                    print(' step: %g,accuracy: %g' % (step,validation_accuracy))
                
                #Add summaries
                if step % log_frequency == 0:
                    train_writer.add_summary(train_summary,step)
                    test_writer.add_summary(test_summary,step)
                step+=1
        print('Training done')
        
        #EVALUATION on test set
        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0
        sess.run(test_iterator.initializer,feed_dict={test_data_placeholder:test_data_sounds, test_labels_placeholder:test_data_labels})
        while evaluated_images != img_number:
            [test_sounds,test_labels] = sess.run(test_batch)
            evaluated_images += len(test_labels)
            test_accuracy_temp = sess.run(accuracy, feed_dict={x: test_sounds, y_: test_labels})
            test_accuracy += test_accuracy_temp
            batch_count += 1
        test_accuracy = test_accuracy / batch_count
        print(test_accuracy)





if __name__ == '__main__':
    tf.app.run(main=main)



