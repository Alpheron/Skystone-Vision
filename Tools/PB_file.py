from common import *
#change the tf example shit to image-tensor
freeze_graph('tf_example', '/workdir/src/Skystone-Vision/Skystone-Rev2/model.config', '/workdir/src/Skystone-Vision/Skystone-Rev2/model_weights/model.ckpt', '/workdir/src/Skystone-Vision/Skystone-Rev2/PB_file/', input_shape=None) 
