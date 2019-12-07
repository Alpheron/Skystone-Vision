from common import *
#change the tf example shit to image-tensor
freeze_graph('image_tensor', '/workdir/src/Skystone-Vision/SkystoneVision/Universal/model.config', '/workdir/src/Skystone-Vision/SkystoneVision/Universal/model_weights/model.ckpt', '/workdir/src/Skystone-Vision/SkystoneVision/image_tensor/PB_file', input_shape=None) 
