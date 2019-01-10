############################################################
#                                                          #
#                SHALLOW NEURAL NETWORK                    #
#                                                          #
############################################################

'''
DATA explained:
length of each train key is 11250
test key: 3750
'data': A list where each entry is an audio segment
'labels': A list where each entry is a 0-based integer label
'track_id': A list where each entry is a unique track id and all the audio segments belonging to the same track 
have the same id, useful for computing the maximum probability and  majority vote metrics.
track id example: [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1 etc]
theres 15 audio segments per track, all have the same label per track
each audio segment has length 20462
'''

##
'''
doing batch norm, getting very low accuracy, try messing around with it, try putting it in the code
and not a function
'''


import sys
import numpy as np
import os
import os.path
import random
import tensorflow as tf
import pickle
from utils import melspectrogram
np.set_printoptions(suppress=True)
np.set_printoptions(1)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_epochs', 100,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('save_model', 1000,
                            'Number of steps between model saves (default: %(default)d)')
tf.app.flags.DEFINE_string('model_type', 'shallow','Type of model used, shallow or deep. (default: %(default)s)')
tf.app.flags.DEFINE_string('learning_type', 'normal','Decaying learning rate or not. (default: %(default)s)')
tf.app.flags.DEFINE_string('initialiser', 'normal','Xavier initialiser or not. (default: %(default)s)')
tf.app.flags.DEFINE_string('batch_norm', 'True','batch norm or not. (default: %(default)s)')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_float('learning_rate', 5e-05, 'Learning rate (default: %(default)d)')
tf.app.flags.DEFINE_string('log_dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')

run_log_dir = os.path.join(FLAGS.log_dir, '{m}_bs_{bs}_{lt}'.format(m=FLAGS.model_type, bs=FLAGS.batch_size, lt=FLAGS.learning_type))

'''
Find one or more examples that are incorrectly classified by the raw measure and
  are corrected by max or major. 

Find at least one failure case where the correct class is the second/third class in the predictions. 
Find at least one failure case where the confidence of the correct class is low. 
For each case, discuss whether you can explain the failure. 
'''




def batch_this(sounds,labels,batch_size,n_sounds,repeat=0,track_id=0,track_tag=0):
    #n_sounds = len(sounds)
    #labels = tf.constant(labels)
    #sounds = tf.constant(sounds)
    if track_id==0:
        dataset = tf.contrib.data.Dataset.from_tensor_slices((sounds,labels))
    elif track_tag==0:
        dataset = tf.contrib.data.Dataset.from_tensor_slices((sounds,labels,track_id))
    else:
        dataset = tf.contrib.data.Dataset.from_tensor_slices((sounds,labels,track_id,track_tag))
    dataset = dataset.shuffle(buffer_size=n_sounds) 
    if repeat:
        dataset = dataset.batch(batch_size).repeat()
    else:
        dataset = dataset.batch(batch_size)#.repeat()
    
    iterator = dataset.make_initializable_iterator()#makes it lag
    return iterator

def relu_norm(x,is_training):
    if FLAGS.batch_norm == 'True':
        return tf.nn.relu(tf.layers.batch_normalization(x,training=is_training))
    elif FLAGS.batch_norm == 'False':
        return tf.nn.relu(x)
def shallownn(x,is_training):
    if FLAGS.initialiser == 'xavier':
        xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True)
    elif FLAGS.initialiser == 'normal':
        xavier_initializer = None
    x = tf.reshape(x, [-1,80,80,1])
    #if is_training == 1:
     #   x_image = tf.map_fn(tf.image.random_flip_left_right,x_image)
    #img_summary = tf.summary.image('Input_images', x_image)
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=16,
        kernel_size=[10,23],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        name='conv1'
    )
    
    conv1 = relu_norm(conv1,is_training)
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[1, 20],
        strides=[1,20],
        name='pool1'
    )
    pool1 = tf.reshape(pool1, [-1,5120])
    conv2 = tf.layers.conv2d(
        inputs=x,
        filters=16,
        kernel_size=[21,20],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        name='conv2'
    )
    conv2 = relu_norm(conv2,is_training)
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[20,1],
        strides=[20,1],
        name='pool2'
    )
    pool2 = tf.reshape(pool2, [-1,5120])
    cnn_out = tf.concat([pool1,pool2],1)
    cnn_out = tf.layers.dropout(cnn_out,rate=0.1)
    fc1 = tf.layers.dense(cnn_out,units=200,activation=tf.nn.relu)
    fcy = tf.layers.dense(fc1,units=10)  
    return fcy

def deepnn(x,is_training):
    x = tf.reshape(x, [-1,80,80,1])
    #Pipeline 1
    #Layer 1
    conv1_1 = tf.layers.conv2d(
        inputs=x,
        filters=16,
        kernel_size=[10,23],
        padding='same',
        use_bias=False,
        #kernel_initializer=xavier_initializer,
        name='conv1_1'
    )
    conv1_1 = tf.nn.relu(conv1_1)
    pool1_1 = tf.layers.max_pooling2d(
        inputs=conv1_1,
        pool_size=[2,2],
        strides=2,
        name='pool1_1'
    )
    #Layer 2
    conv1_2 = tf.layers.conv2d(
        inputs=pool1_1,
        filters=32,
        kernel_size=[5,11],
        padding='same',
        use_bias=False,
        #kernel_initializer=xavier_initializer,
        name='conv1_2'
    )
    conv1_2 = tf.nn.relu(conv1_2)
    pool1_2 = tf.layers.max_pooling2d(
        inputs=conv1_2,
        pool_size=[2,2],
        strides=2,
        name='pool1_2'
    )
    #Layer 3
    conv1_3 = tf.layers.conv2d(
        inputs=pool1_2,
        filters=64,
        kernel_size=[3,5],
        padding='same',
        use_bias=False,
        #kernel_initializer=xavier_initializer,
        name='conv1_3'
    )
    conv1_3 = tf.nn.relu(conv1_3)
    pool1_3 = tf.layers.max_pooling2d(
        inputs=conv1_3,
        pool_size=[2,2],
        strides=2,
        name='pool1_3'
    )
    #Layer 4
    conv1_4 = tf.layers.conv2d(
        inputs=pool1_3,
        filters=128,
        kernel_size=[2,4],
        padding='same',
        use_bias=False,
        #kernel_initializer=xavier_initializer,
        name='conv1_4'
    )
    conv1_4 = tf.nn.relu(conv1_4)
    pool1_4 = tf.layers.max_pooling2d(
        inputs=conv1_4,
        pool_size=[1,5],
        strides=[1,5],
        name='pool1_4'
    )
    pool1_4 = tf.reshape(pool1_4, [-1,2560])
    ########
    #Pipeline 2
    #Layer 1
    conv2_1 = tf.layers.conv2d(
        inputs=x,
        filters=16,
        kernel_size=[10,23],
        padding='same',
        use_bias=False,
        #kernel_initializer=xavier_initializer,
        name='conv2_1'
    )
    conv2_1 = tf.nn.relu(conv2_1)
    pool2_1 = tf.layers.max_pooling2d(
        inputs=conv2_1,
        pool_size=[2,2],
        strides=2,
        name='pool2_1'
    )
    #Layer 2
    conv2_2 = tf.layers.conv2d(
        inputs=pool2_1,
        filters=32,
        kernel_size=[5,11],
        padding='same',
        use_bias=False,
        #kernel_initializer=xavier_initializer,
        name='conv2_2'
    )
    conv2_2 = tf.nn.relu(conv2_2)
    pool2_2 = tf.layers.max_pooling2d(
        inputs=conv2_2,
        pool_size=[2,2],
        strides=2,
        name='pool2_2'
    )
    #Layer 3
    conv2_3 = tf.layers.conv2d(
        inputs=pool2_2,
        filters=64,
        kernel_size=[3,5],
        padding='same',
        use_bias=False,
        #kernel_initializer=xavier_initializer,
        name='conv2_3'
    )
    conv2_3 = tf.nn.relu(conv2_3)
    pool2_3 = tf.layers.max_pooling2d(
        inputs=conv2_3,
        pool_size=[2,2],
        strides=2,
        name='pool2_3'
    )
    #Layer 4
    conv2_4 = tf.layers.conv2d(
        inputs=pool2_3,
        filters=128,
        kernel_size=[2,4],
        padding='same',
        use_bias=False,
        #kernel_initializer=xavier_initializer,
        name='conv2_4'
    )
    conv2_4 = tf.nn.relu(conv2_4)
    pool2_4 = tf.layers.max_pooling2d(
        inputs=conv2_4,
        pool_size=[1,5],
        strides=[1,5],
        name='pool2_4'
    )
    pool2_4 = tf.reshape(pool2_4, [-1,2560])
    #######
    #Merge
    cnn_out = tf.concat([pool1_4,pool2_4],1)
    cnn_out = tf.layers.dropout(cnn_out,rate=0.25)
    #fc1 = tf.layers.dense(cnn_out,units=200)#,activation=tf.nn.relu)
    fc1 = tf.layers.dense(cnn_out,units=200,activation=tf.nn.relu) 
    fcy = tf.layers.dense(fc1,units=10) 
    return fcy

###############

def main(_):
    tf.reset_default_graph()
    # Import data
    train_data_sounds = np.load('data/train_data_postmel.npy')
    train_data_labels = np.load('data/train_labels.npy')
    test_data_sounds = np.load('data/test_data_postmel.npy')
    test_data_labels_og = np.load('data/test_labels.npy')
    test_data_id = np.load('data/test_id.npy')
    test_data_tag = np.arange(3750)
    img_number = len(test_data_labels_og)
    print('Data Loaded')
    log_per_epoch = int(1000/FLAGS.max_epochs)
    log_frequency = len(train_data_labels)/FLAGS.batch_size/log_per_epoch
    if log_frequency == 0 :
        log_frequency+=1
    train_data_placeholder = tf.placeholder(tf.float32, [None, 80, 80])
    train_labels_placeholder = tf.placeholder(tf.int32, [None])
    test_data_placeholder = tf.placeholder(tf.float32, [None, 80, 80])
    test_labels_placeholder = tf.placeholder(tf.int32, [None])
    test_id_placeholder = tf.placeholder(tf.int32, [None])
    test_tag_placeholder = tf.placeholder(tf.int32, [None])

    #Train split
    random_n = np.random.randint(0,100000)
    random_n = 0
    np.random.seed(random_n)
    #np.random.shuffle(train_data_sounds)
    np.random.seed(random_n)
    #np.random.shuffle(train_data_labels)
    n_sounds = len(train_data_labels)
    train_iterator = batch_this(train_data_placeholder,train_labels_placeholder,FLAGS.batch_size,n_sounds)
    train_batch = train_iterator.get_next()
    random_n = np.random.randint(0,100000)
    random_n = 0
    #Test split
    np.random.seed(random_n)
    #np.random.shuffle(test_data_sounds)
    np.random.seed(random_n)
    #test_data_labels = np.random.permutation(test_data_labels_og)
    test_data_labels = test_data_labels_og
    #test_data_labels = test_data_labels_og
    np.random.seed(random_n)
    #np.random.shuffle(test_data_id)
    np.random.seed(random_n)
    #np.random.shuffle(test_data_tag)
    test_iterator = batch_this(test_data_placeholder,test_labels_placeholder,FLAGS.batch_size,len(test_data_labels),1)
    test_batch = test_iterator.get_next()
    test_iterator_plus = batch_this(test_data_placeholder,test_labels_placeholder,FLAGS.batch_size,len(test_data_labels),1,test_id_placeholder,test_tag_placeholder)
    test_batch_plus = test_iterator_plus.get_next()
    print('All batched')
    
    with tf.variable_scope('inputs'):
        # Create the model
        x = tf.placeholder(tf.float32, [None,80,80])
        # Define loss and optimizer
        y_ = tf.placeholder(tf.int32, [None])
    is_training = tf.placeholder(tf.bool)
    ###MODEL USED
    if FLAGS.model_type == 'shallow':
        y_conv = shallownn(x,is_training)
    elif FLAGS.model_type == 'deep':
        y_conv = deepnn(x,is_training)
    with tf.variable_scope('x_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    
    l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.0001)
    all_weights = tf.trainable_variables()
    regularization_factor = tf.contrib.layers.apply_regularization(l1_regularizer, weights_list= all_weights)
    cross_entropy += regularization_factor 
    if FLAGS.learning_type == 'decay':
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,global_step ,1000,0.8)
        optimiser = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)
    elif FLAGS.learning_type == 'normal':
        optimiser = tf.train.AdamOptimizer(FLAGS.learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08).minimize(cross_entropy)
    accuracy = tf.equal(tf.argmax(y_conv,1),tf.cast(y_, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

    #Summaries
    loss_summary = tf.summary.scalar('Loss', cross_entropy)
    acc_summary = tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge([loss_summary,acc_summary])
    


    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(run_log_dir + '_train', sess.graph,flush_secs=5)
        test_writer = tf.summary.FileWriter(run_log_dir + '_test', sess.graph,flush_secs=5)
        sess.run(tf.global_variables_initializer())
        step = 0

        for epoch in range(FLAGS.max_epochs):
            # Training: Backpropagation using train set
            sess.run(train_iterator.initializer,feed_dict={train_data_placeholder:train_data_sounds,train_labels_placeholder:train_data_labels})
            sess.run(test_iterator.initializer,feed_dict={test_data_placeholder:test_data_sounds,test_labels_placeholder:test_data_labels})
            print('Epoch:',epoch)
            while True:
                try:
                    [train_sounds,train_labels] = sess.run(train_batch)
                    _,train_summary = sess.run([optimiser,merged], feed_dict={x: train_sounds, y_: train_labels,is_training:True}) 
                except tf.errors.OutOfRangeError:
                    break
                #Accuracy
                [test_sounds,test_labels] = sess.run(test_batch)
                validation_accuracy,test_summary = sess.run([accuracy,merged], feed_dict={x: test_sounds, y_: test_labels, is_training:False})
                #if step % 170 == 0:
                print(' step: %g,accuracy: %g' % (step,validation_accuracy))
                
                #Add summaries
                '''
                if step % log_frequency == 0:
                    train_writer.add_summary(train_summary,step)
                    test_writer.add_summary(test_summary,step)
                '''
                step+=1
        #print('Training done')
        
        ############################################################
        #                                                          #
        #                EVALUATION                                #
        #                                                          #
        ############################################################
        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0
        prob_array_max_prob = np.zeros([250,10])
        prob_array_maj_vote = np.zeros([250,10])
        segment_predic = np.zeros([250,15])
        segment_tag = np.zeros([250,15])
        sess.run(test_iterator_plus.initializer,feed_dict={test_data_placeholder:test_data_sounds,
                                                           test_labels_placeholder:test_data_labels,test_id_placeholder:test_data_id,test_tag_placeholder:test_data_tag})
        while evaluated_images != img_number:
            [test_sounds,test_labels,test_id,test_tag] = sess.run(test_batch_plus)
            #Outputs of cnn for current batch
            #Get pure predictions for current batch and softmax them
            current_proba = sess.run(y_conv,feed_dict={x: test_sounds, y_: test_labels, is_training:False})
            current_proba = sess.run(tf.nn.softmax(current_proba,1))
            current_argmax = sess.run(tf.argmax(current_proba,1))
                
            #loop through id of batch and for each track, find prediction and add +1 to the column of class
            for idx,current_id in enumerate(test_id):
                #add softmax values for max prob
                prob_array_max_prob[current_id] += current_proba[idx,:]
                #add 1 at argmax value for majority vote
                prob_array_maj_vote[current_id,current_argmax[idx]] += 1

                segment_predic[current_id,test_tag[idx]%15] += current_argmax[idx]
                segment_tag[current_id,test_tag[idx]%15] += test_tag[idx]
            evaluated_images += len(test_labels)
            #add accuracy for raw proba
            test_accuracy_temp = sess.run(accuracy, feed_dict={x: test_sounds, y_: test_labels, is_training:False})
            test_accuracy += test_accuracy_temp
            batch_count += 1

        test_accuracy = test_accuracy / batch_count
        #print('test set: accuracy on test set: %0.3f' % (test_accuracy))
        #prediction per track for max proba
        max_proba = sess.run(tf.argmax(prob_array_max_prob,1))  
        
        #get 1st labels for all consecutive segments of same track  
        label_per_track = []
        for i in range(int(len(test_data_labels_og)/15)):
            label_per_track.append(test_data_labels_og[i*15])
        for i in range(len(max_proba)):
            if max_proba[i] == label_per_track[i]:
                track_chosen = i
                break
        max_proba_accuracy = sess.run(tf.reduce_mean(tf.cast(tf.equal(max_proba,label_per_track),tf.float32)))
        #print('test set: max proba accuracy on test set: %0.3f' % (max_proba_accuracy)) 
        majority_vote = sess.run(tf.argmax(prob_array_maj_vote,1))
        majority_vote_accuracy = sess.run(tf.reduce_mean(tf.cast(tf.equal(majority_vote,label_per_track),tf.float32)))
        #print('test set: Majority vote accuracy on test set: %0.3f' % (majority_vote_accuracy)) 
        print(test_accuracy,max_proba_accuracy,majority_vote_accuracy)
        #print(FLAGS.initialiser)
        #print('-------')
        '''
        print(prob_array_max_prob[track_chosen])
        print(prob_array_maj_vote[track_chosen])
        print(max_proba[track_chosen])
        print(label_per_track[track_chosen])
        print(segment_predic[track_chosen])
        print(segment_tag[track_chosen])
        '''
        #print(FLAGS.learning_type)
        #print(FLAGS.initialiser)
        print(FLAGS.batch_norm)

    #print('done')



if __name__ == '__main__':
    tf.app.run(main=main)



#normal:
#0.68, 0.82, 0.8
#0.65, 0.81, 0.8
#0.67 0.82 0.80

#xavier: