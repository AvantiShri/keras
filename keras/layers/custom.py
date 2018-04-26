from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .. import backend as K
from .. import activations
from .. import initializers
from .. import regularizers
from .. import constraints
from ..engine.base_layer import Layer
from ..engine.base_layer import InputSpec
from ..utils import conv_utils


class SwitchReLU(Layer):
    """
    """

    def __init__(self, unshared_axes,
                 alpha_initializer='zeros',
                 sigmoid_multiplier=10,
                 alpha_regularizer=None,
                 alpha_constraint=None,
                 **kwargs):
        super(SwitchReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.sigmoid_multiplier = sigmoid_multiplier
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        if not isinstance(unshared_axes, (list, tuple)):
            self.unshared_axes = [unshared_axes]
        else:
            self.unshared_axes = list(unshared_axes)

    def build(self, input_shape):
        param_shape = [1 for x in range(len(input_shape)-1)] #drop batch axis
        self.param_broadcast = [True] * len(param_shape)
        for i in self.unshared_axes:
            assert i!=0, "batch axis (0) invalid as an unshared axis"
            pos_i = i if i > 0 else (len(input_shape)+i)
            param_shape[pos_i - 1] = input_shape[pos_i]
            self.param_broadcast[pos_i - 1] = False
        self.alpha = self.add_weight(shape=param_shape,
                                     name='alpha',
                                     initializer=self.alpha_initializer,
                                     regularizer=self.alpha_regularizer,
                                     constraint=self.alpha_constraint)
        # Set input spec
        axes = {}
        for i in self.unshared_axes:
            pos_i = i if i > 0 else (len(input_shape)+i)
            axes[pos_i] = input_shape[pos_i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs, mask=None):
        sigmoid_input = self.sigmoid_multiplier*(inputs + self.alpha)
        return K.relu(inputs)*K.sigmoid(sigmoid_input)
         
    def get_config(self):
        config = {
            'sigmoid_multiplier': self.sigmoid_multiplier,
            'alpha_initializer': initializers.serialize(self.alpha_initializer),
            'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
            'alpha_constraint': constraints.serialize(self.alpha_constraint),
            'unshared_axes': self.unshared_axes
        }
        base_config = super(SwitchReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class GluConv1D(Layer):
    """
    """

    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 apply_relu_to_linear_bit=False,
                 **kwargs):
        super(GluConv1D, self).__init__(**kwargs)
        rank = 1
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.apply_relu_to_linear_bit = apply_relu_to_linear_bit
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernelA = self.add_weight(shape=kernel_shape,
                                       initializer=self.kernel_initializer,
                                       name='kernelA',
                                       regularizer=self.kernel_regularizer,
                                       constraint=self.kernel_constraint)

        self.kernelB = self.add_weight(shape=kernel_shape,
                                       initializer=self.kernel_initializer,
                                       name='kernelB',
                                       regularizer=self.kernel_regularizer,
                                       constraint=self.kernel_constraint)
        if self.use_bias:
            self.biasA = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='biasA',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.biasB = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='biasB',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):

        gate_outputs = K.conv1d(
            inputs,
            self.kernelA,
            strides=self.strides[0],
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate[0])

        signal_outputs = K.conv1d(
            inputs,
            self.kernelB,
            strides=self.strides[0],
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate[0])

        if self.use_bias:
            gate_outputs = K.bias_add(
                gate_outputs,
                self.biasA,
                data_format=self.data_format)
            signal_outputs = K.bias_add(
                signal_outputs,
                self.biasB,
                data_format=self.data_format) 

        if (self.apply_relu_to_linear_bit):
            return K.sigmoid(gate_outputs)*K.relu(signal_outputs)
        else:
            return K.sigmoid(gate_outputs)*signal_outputs

    #copied from _Conv
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'apply_relu_to_linear_bit': self.apply_relu_to_linear_bit
        }
        base_config = super(GluConv1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
