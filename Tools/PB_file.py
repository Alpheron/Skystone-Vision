from common import *
#change the tf example shit to image-tensor
freeze_graph('tf_example', '/workdir/src/Skystone-Vision/Skystone-Rev3/Universal/model.config', '/workdir/src/Skystone-Vision/Skystone-Rev3/Universal/model_weights/model.ckpt', '/workdir/src/Skystone-Vision/Skystone-Rev3/tf_example/PB_file', input_shape=None) 
