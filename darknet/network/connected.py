class Connected(Dense):
    """
    Darknet "connected" layer. Main difference vs. keras Dense layer is that 
    input also becomes flatten.
    The same as 
    
    ```
    def get_connected(params):
        activation = get_activation(params.get('activation', "linear"))
        def _connected(x):
            y = Flatten()(x)
            return Dense(params.get('output', 1), activation=activation)(y)
        
        return Lambda(_connected)
    ```
    
    - but also has weights.
    """
    def __init__(self, **kwargs):
        super(Connected, self).__init__(**kwargs)
    
    def build(self, input_shape):
        densed_shape = (input_shape[0], np.prod(input_shape[1:]))
        super(Connected, self).build(densed_shape) 
        
    
    def call(self, x, training=None):
        inputs = K.batch_flatten(x)
        return super(Connected, self).call(inputs)
        
        
    def compute_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise ValueError('The shape of the input to "Connected" '
                             'is not fully defined '
                             '(got ' + str(input_shape[1:]) + '. '
                             'Make sure to pass a complete "input_shape" '
                             'or "batch_input_shape" argument to the first '
                             'layer in your model.')
        return (input_shape[0], np.prod(input_shape[1:]))
        