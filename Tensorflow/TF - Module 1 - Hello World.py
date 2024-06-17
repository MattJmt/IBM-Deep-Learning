import tensorflow as tf
# Build a graph
graph1 = tf.Graph()
with graph1.as_default():
    a = tf.constant([2], name = 'constant_a')
    b = tf.constant([3], name = 'constant_b')

# Printing the value of a
sess = tf.compat.v1.Session( target='', graph=graph1, config=None)
result = sess.run(a)
print(result)
sess.close()

# DEFINING MULTIPLE DIMENSION ARRAYS
graph2 = tf.Graph()
with graph2.as_default():
    Scalar = tf.constant(2)
    Vector = tf.constant([5,6,2])
    Matrix = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
    Tensor = tf.constant( [ [[1,2,3],[2,3,4],[3,4,5]] , [[4,5,6],[5,6,7],[6,7,8]] , [[7,8,9],[8,9,10],[9,10,11]] ] )
with tf.Session(graph = graph2) as sess:
    result = sess.run(Scalar)
    print ("Scalar (1 entry):\n %s \n" % result)
    result = sess.run(Vector)
    print ("Vector (3 entries) :\n %s \n" % result)
    result = sess.run(Matrix)
    print ("Matrix (3x3 entries):\n %s \n" % result)
    result = sess.run(Tensor)
    print ("Tensor (3x3x3 entries) :\n %s \n" % result)
Scalar.shape
Tensor.shape
graph3 = tf.Graph()
with graph3.as_default():
    Matrix_one = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
    Matrix_two = tf.constant([[2,2,2],[2,2,2],[2,2,2]])

    add_1_operation = tf.add(Matrix_one, Matrix_two)
    add_2_operation = Matrix_one + Matrix_two

with tf.Session(graph =graph3) as sess:
    result = sess.run(add_1_operation)
    print ("Defined using tensorflow function :")
    print(result)
    result = sess.run(add_2_operation)
    print ("Defined using normal expressions :")
    print(result)
graph4 = tf.Graph()
with graph4.as_default():
        Matrix_one = tf.constant([[2, 3], [3, 4]])
        Matrix_two = tf.constant([[2, 3], [3, 4]])

        mul_operation = tf.matmul(Matrix_one, Matrix_two)

with tf.Session(graph=graph4) as sess:
        result = sess.run(mul_operation)
        print("Defined using tensorflow function :")
        print(result)

# Variables
v = tf.Variable(0)
update = tf.assign(v, v+1)
init_op = tf.global_variables_initializer() #Variables must be initialized by running an initialization operation after having launched the graph.  We first have to add the initialization operation to the graph:
with tf.Session() as session:
    session.run(init_op)
    print(session.run(v))
    for _ in range(3):
        session.run(update)
        print(session.run(v))

# PLACEHOLDERS
a = tf.placeholder(tf.float32)
b = a * 2
with tf.Session() as sess:
    result = sess.run(b,feed_dict={a:3.5})
    print (result)
dictionary={a: [ [ [1,2,3],[4,5,6],[7,8,9],[10,11,12] ] , [ [13,14,15],[16,17,18],[19,20,21],[22,23,24] ] ] }

with tf.Session() as sess:
    result = sess.run(b,feed_dict=dictionary)
    print (result)

# Operations
graph5 = tf.Graph()
with graph5.as_default():
    a = tf.constant([5])
    b = tf.constant([2])
    c = tf.add(a,b)
    d = tf.subtract(a,b)

with tf.Session(graph = graph5) as sess:
    result = sess.run(c)
    print ('c =: %s' % result)
    result = sess.run(d)
    print ('d =: %s' % result)