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
                 **kwargs):
        super(GluConv1D, self).__init__(**kwargs)
        self.rank = 1
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

        return K.sigmoid(gate_outputs)*(K.relu(signal_outputs)+1)

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
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(_Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
