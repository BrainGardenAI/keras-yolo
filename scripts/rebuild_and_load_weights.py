"""Script to check darknet weights reading functionality.

Script runs distutils to rebuild sources and tries to read weights from darknet weights file.

This script should be run from the main project directory.
"""

def rebuild_source():
    import subprocess, os, sys
    returncode = subprocess.call([
            sys.executable, "setup.py", "build_ext", "--inplace"
        ], cwd=os.getcwd())
    print("Finished rebuild with %s exit code" % returncode)
    return returncode



def check_weights_loading(model_file_name, weight_file_name):
    import os, sys
    #if not os.path.exists(model_file_name):
    #    print("darknet model config file %s not found" % model_file)
    #    return
    if not os.path.exists(weight_file_name):
        print("darknet weight file %s could not be found" % weight_file_name)
        return
    sys.path.insert(0, os.getcwd())
    from darknet.network import buildYoloModel
    from darknet.weights_reader import read_file
    
    model, layer_names = buildYoloModel(model_file_name)
    #print([(x.shape)  for x in model.get_weights()])
    for layer_type, layer in zip(layer_names, model.layers):
        print(layer_type, layer.name)
        print [ x.shape for x in layer.get_weights()]
        print(layer.output_shape)
    print("-----\nstart actual reading\n-----\n")
    read_file(weight_file_name, model, layer_names)
    pass


if __name__ == "__main__":
    resultcode = rebuild_source()
    if resultcode:
        print("Rebuild failed, exiting")
        exit(1)
    check_weights_loading("cfg/yolov1/tiny-yolo.cfg", "data/tiny-yolo.weights")
    pass
