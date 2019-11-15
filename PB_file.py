from common.py import freeze_graph

freeze_graph(tf_example, '/workdir/src/Skystone-Vision/SkystoneVision/model.config', '/workdir/src/Skystone-Vision/SkystoneVision', '/workdir/src/Skystone-Vision/Output', input_shape=None) 
