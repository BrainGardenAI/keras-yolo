import numpy as np
cimport numpy as np

from libc.stdio cimport FILE, fopen, fwrite, fscanf, fclose, fseek, SEEK_END, fread, ftell, stdout, stderr

#cdef extern from "weights_loader.h":
#  cdef void read_header_(FILE* fp) 


cdef void read_header(FILE* fp):
    cdef int major, minor, revision, seen
    fread(&major, sizeof(int), 1, fp)
    fread(&minor, sizeof(int), 1, fp)
    fread(&revision, sizeof(int), 1, fp)
    fread(&seen, sizeof(int), 1, fp)
    cdef int transpose = (major > 1000) or (minor > 1000);
    print(major, minor, revision, seen, transpose)


cdef get_channels(layer):
    if layer.data_format == 'channels_last':
        return layer.input_shape[-1]
    return layer.input_shape[0]
    

cdef void read_convolutional_weights(FILE*fp, layer):
    channels = get_channels(layer)

    cdef np.ndarray[np.float32_t, ndim=1, mode='c'] biases = np.zeros((layer.filters,), dtype=np.float32)
    fread(&biases[0], sizeof(np.float32_t), layer.filters, fp)
    
    cdef int num = layer.filters*channels*layer.kernel_size[0]*layer.kernel_size[0]
    
    cdef np.ndarray[np.float32_t, ndim=1, mode='c'] scales = np.zeros((layer.filters,), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1, mode='c'] rolling_mean = np.zeros((layer.filters,), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1, mode='c'] rolling_variance = np.zeros((layer.filters,), dtype=np.float32)

    if 1: #there should be batch_normalize
        fread(&scales[0], sizeof(np.float32_t), layer.filters, fp)
        fread(&rolling_mean[0], sizeof(np.float32_t), layer.filters, fp)
        fread(&rolling_variance[0], sizeof(np.float32_t), layer.filters, fp)

    cdef np.ndarray[np.float32_t, ndim=4, mode='c'] weights = np.zeros(layer.kernel_size + (channels, layer.filters), dtype=np.float32)
    fread(&weights[0,0,0,0], sizeof(np.float32_t), num, fp)
    layer.set_weights((weights, biases))
    
    
cdef void read_connected_weights(FILE*fp, layer):
    print(layer.input_shape) 
    output_size = layer.output_shape[1]
    cdef np.ndarray[np.float32_t, ndim=1, mode='c'] biases = np.zeros((output_size,), dtype=np.float32)
    fread(&biases[0], sizeof(np.float32_t), output_size, fp)
    cdef int num = output_size * np.prod(layer.input_shape[1:])
    #print(num, tuple(layer.input_shape[1:]) + (output_size,))
    cdef np.ndarray[np.float32_t, ndim=2, mode='fortran'] weights = np.zeros((np.prod(layer.input_shape[1:]), 
        output_size), dtype=np.float32, order='F')
    fread(&weights[0,0], sizeof(np.float32_t), num, fp)
    cdef np.ndarray[np.float32_t, ndim=1, mode='c'] scales = np.zeros((output_size,), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1, mode='c'] rolling_mean = np.zeros((output_size,), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1, mode='c'] rolling_variance = np.zeros((output_size,), dtype=np.float32)
    if 1:
        fread(&scales[0], sizeof(np.float32_t), output_size, fp)
        fread(&rolling_mean[0], sizeof(np.float32_t), output_size, fp)
        fread(&rolling_variance[0], sizeof(np.float32_t), output_size, fp)
    layer.set_weights((weights, biases))
    #fread(l.biases, sizeof(float), l.outputs, fp);
    #fread(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    #if(transpose){
    #    transpose_matrix(l.weights, l.inputs, l.outputs);
    #}
    #if (l.batch_normalize && (!l.dontloadscales)){
    #    fread(l.scales, sizeof(float), l.outputs, fp);
    #    fread(l.rolling_mean, sizeof(float), l.outputs, fp);
    #    fread(l.rolling_variance, sizeof(float), l.outputs, fp);
    #    //printf("Scales: %f mean %f variance\n", mean_array(l.scales, l.outputs), variance_array(l.scales, l.outputs));
    #    
    #}
    pass
    
    
cdef void read_local_weights(FILE * fp, l):
    # not implemented
    print(l.output_shape, "<---")
    cdef int locations = l.output_shape[1]*l.output_shape[2]
    #  int locations = l.out_w*l.out_h;
    #    int size = l.size*l.size*l.c*l.n*locations;
    #    fread(l.biases, sizeof(float), l.outputs, fp);
    #    fread(l.weights, sizeof(float), size, fp);
    pass
  
  
cpdef void read_file(filename, model, layer_names):
    print(1)
    cdef FILE* ptr = fopen(filename, "rb")
    read_header(ptr)
    for layer_type, layer in zip(layer_names, model.layers):
          print(layer_type)
          print [ x.shape for x in layer.get_weights()]
          print("processing %s"%layer_type)
          if layer_type == "convolutional":
              print("processing convolutional layer %s\n"% layer.name)
              read_convolutional_weights(ptr, layer)
          if layer_type == "connected":
              print("processing connected layer %s\n" %layer.name)
              read_connected_weights(ptr, layer)
              #return
          if layer_type == "local":
              print("processing local layer %s\n" % layer.name)
              #read_local_weights(ptr, layer)
              #return
    pos = ftell(ptr)
    fseek(ptr, 0, SEEK_END)
    assert(ftell(ptr) == pos, "File wasn't processed completely")
    pass
