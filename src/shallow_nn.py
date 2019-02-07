import tensorflow as tf
def shallow(x,initialiser):
    #Shallow network with 2 parallel sets of convolutional and max pool layers
    if initialiser == 'xavier':
        my_initializer = tf.contrib.layers.xavier_initializer(uniform=True)
    elif initialiser == 'normal':
        my_initializer = None
    x = tf.reshape(x, [-1,80,80,1])
    #Conv layer 1
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=16,
        kernel_size=[10,23],
        padding='same',
        use_bias=False,
        kernel_initializer=my_initializer,
        name='conv1'
    )
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[1, 20],
        strides=[1,20],
        name='pool1'
    )
    pool1 = tf.reshape(pool1, [-1,5120])
    #Conv layer 2
    conv2 = tf.layers.conv2d(
        inputs=x,
        filters=16,
        kernel_size=[21,20],
        padding='same',
        use_bias=False,
        kernel_initializer=my_initializer,
        name='conv2'
    )
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[20,1],
        strides=[20,1],
        name='pool2'
    )
    pool2 = tf.reshape(pool2, [-1,5120])
    #Merge
    cnn_out = tf.concat([pool1,pool2],1)
    cnn_out = tf.layers.dropout(cnn_out,rate=0.1)
    #Dense layers
    fc1 = tf.layers.dense(cnn_out,units=200,activation=tf.nn.relu)
    fcy = tf.layers.dense(fc1,units=10)  
    return fcy