from common import *
#change the tf example shit to image-tensor
freeze_graph('image_tensor', '/workdir/src/Skystone-Vision/Skystone-Rev6/Universal/model.config', '/workdir/src/Skystone-Vision/Skystone-Rev6/Universal/model_weights/model.ckpt', '/workdir/src/Skystone-Vision/Skystone-Rev6/image_tensor/PB_file', input_shape=None) 
