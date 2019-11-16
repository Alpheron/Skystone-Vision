from common import *

freeze_graph('tf_example', '/workdir/src/Skystone-Vision/SkystoneVision/model.config', '/workdir/src/Skystone-Vision/SkystoneVision/model_weights/model.ckpt', '/workdir/src/Skystone-Vision/Output', input_shape=None) 
