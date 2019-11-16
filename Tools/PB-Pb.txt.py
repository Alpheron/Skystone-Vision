import tensorflow.compat.v1 as tf

#from google.protobuf import text_format
from tensorflow.python.platform import gfile

def converter(filename):
  with gfile.FastGFile(filename,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    tf.train.write_graph(graph_def, 'pbtxt/', 'protobuf.pbtxt', as_text=True)
    print(graph_def)
  return



# and then a new file will be made in pbtxt directory.

converter('/Users/Tinku/Skystone-Vision/SkystoneVision/Output/model.pb')
