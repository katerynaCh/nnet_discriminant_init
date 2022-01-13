# Copyright 2021 Kateryna Chumachenko
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# THis code is based on the following:

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Normalization layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tensorflow.python.compat import compat
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables
#from tensorflow.python.platform import device_context
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
import tensorflow as tf

import sys
class CBatchNormalizationBase(Layer):
  r"""Normalize and scale inputs or activations. (Ioffe and Szegedy, 2014).
  Normalize the activations of the previous layer at each batch,
  i.e. applies a transformation that maintains the mean activation
  close to 0 and the activation standard deviation close to 1.
  Batch normalization differs from other layers in several key aspects:
  1) Adding BatchNormalization with `training=True` to a model causes the
  result of one example to depend on the contents of all other examples in a
  minibatch. Be careful when padding batches or masking examples, as these can
  change the minibatch statistics and affect other examples.
  2) Updates to the weights (moving statistics) are based on the forward pass
  of a model rather than the result of gradient computations.
  3) When performing inference using a model containing batch normalization, it
  is generally (though not always) desirable to use accumulated statistics
  rather than mini-batch statistics. This is accomplished by passing
  `training=False` when calling the model, or using `model.predict`.
  Arguments:
    axis: Integer, the axis that should be normalized
      (typically the features axis).
      For instance, after a `Conv2D` layer with
      `data_format="channels_first"`,
      set `axis=1` in `BatchNormalization`.
    momentum: Momentum for the moving average.
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor.
      If False, `beta` is ignored.
    scale: If True, multiply by `gamma`.
      If False, `gamma` is not used.
      When the next layer is linear (also e.g. `nn.relu`),
      this can be disabled since the scaling
      will be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    moving_mean_initializer: Initializer for the moving mean.
    moving_variance_initializer: Initializer for the moving variance.
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    beta_constraint: Optional constraint for the beta weight.
    gamma_constraint: Optional constraint for the gamma weight.
    renorm: Whether to use Batch Renormalization
      (https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    renorm_momentum: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `momentum` is still applied
      to get the means and variances for inference.
    fused: if `True`, use a faster, fused implementation, or raise a ValueError
      if the fused implementation cannot be used. If `None`, use the faster
      implementation if possible. If False, do not used the fused
      implementation.
    trainable: Boolean, if `True` the variables will be marked as trainable.
    virtual_batch_size: An `int`. By default, `virtual_batch_size` is `None`,
      which means batch normalization is performed across the whole batch. When
      `virtual_batch_size` is not `None`, instead perform "Ghost Batch
      Normalization", which creates virtual sub-batches which are each
      normalized separately (with shared gamma, beta, and moving statistics).
      Must divide the actual batch size during execution.
    adjustment: A function taking the `Tensor` containing the (dynamic) shape of
      the input tensor and returning a pair (scale, bias) to apply to the
      normalized values (before gamma and beta), only during training. For
      example, if axis==-1,
        `adjustment = lambda shape: (
          tf.random.uniform(shape[-1:], 0.93, 1.07),
          tf.random.uniform(shape[-1:], -0.1, 0.1))`
      will scale the normalized value by up to 7% up or down, then shift the
      result by up to 0.1 (with independent scaling and bias for each feature
      but shared across all examples), and finally apply gamma and/or beta. If
      `None`, no adjustment is applied. Cannot be specified if
      virtual_batch_size is specified.
  Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode.
      - `training=True`: The layer will normalize its inputs using the
        mean and variance of the current batch of inputs.
      - `training=False`: The layer will normalize its inputs using the
        mean and variance of its moving statistics, learned during training.
  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.
  Output shape:
    Same shape as input.
  {{TRAINABLE_ATTRIBUTE_NOTE}}
  Normalization equations:
    Consider the intermediate activations \(x\) of a mini-batch of size
    \\(m\\):
    We can compute the mean and variance of the batch
    \\({\mu_B} = \frac{1}{m} \sum_{i=1}^{m} {x_i}\\)
    \\({\sigma_B^2} = \frac{1}{m} \sum_{i=1}^{m} ({x_i} - {\mu_B})^2\\)
    and then compute a normalized \\(x\\), including a small factor
    \\({\epsilon}\\) for numerical stability.
    \\(\hat{x_i} = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}\\)
    And finally \\(\hat{x}\) is linearly transformed by \({\gamma}\\)
    and \\({\beta}\\), which are learned parameters:
    \\({y_i} = {\gamma * \hat{x_i} + \beta}\\)
  References:
  - [Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
  """

  # By default, the base class uses V2 behavior. The BatchNormalization V1
  # subclass sets this to False to use the V1 behavior.
  _USE_V2_BEHAVIOR = True

  def __init__(self,
               filtersize=2,
               axis=-1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               moving_mean_initializer='zeros',
               moving_variance_initializer='ones',
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               renorm=False,
               renorm_clipping=None,
               renorm_momentum=0.99,
               fused=None,
               trainable=True,
               virtual_batch_size=None,
               adjustment=None,
               name=None,
               **kwargs):
    super(CBatchNormalizationBase, self).__init__(name=name, **kwargs)
    if isinstance(axis, (list, tuple)):
      self.axis = axis[:]
    elif isinstance(axis, int):
      self.axis = axis
    else:
      raise TypeError('Expected an int or a list/tuple of ints for the '
                      'argument \'axis\', but received: %r' % axis)
    #tf.print('Initializing', output_stream=sys.stderr)
    self.momentum = momentum
    self.filtersize = filtersize
    self.epsilon = epsilon
    self.center = center
    self.scale = scale
    self.beta_initializer = initializers.get(beta_initializer)
    self.gamma_initializer = initializers.get(gamma_initializer)
    self.moving_mean_initializer = initializers.get(moving_mean_initializer)
    self.moving_variance_initializer = initializers.get(
        moving_variance_initializer)
    self.beta_regularizer = regularizers.get(beta_regularizer)
    self.gamma_regularizer = regularizers.get(gamma_regularizer)
    self.beta_constraint = constraints.get(beta_constraint)
    self.gamma_constraint = constraints.get(gamma_constraint)
    self.renorm = False
    self.virtual_batch_size = None
    self.adjustment = adjustment
    if self._USE_V2_BEHAVIOR:
      if fused:
        self._raise_if_fused_cannot_be_used()
      # We leave fused as None if self._fused_can_be_used()==True, since we
      # still may set it to False in self.build() if the input rank is not 4.
      elif fused is None and not self._fused_can_be_used():
        fused = False
    elif fused is None:
      fused = True
    self.supports_masking = True

    self.fused = False
    self._bessels_correction_test_only = True
    self._trainable_var = None
    self.trainable = trainable

  @property
  def trainable(self):
    return self._trainable

  @trainable.setter
  def trainable(self, value):
    self._trainable = value
    if self._trainable_var is not None:
      self._trainable_var.update_value(value)

  def _get_trainable_var(self):
    if self._trainable_var is None:
      self._trainable_var = K.freezable_variable(
          self._trainable, name=self.name + '_trainable')
    return self._trainable_var

  @property
  def _param_dtype(self):
    # Raise parameters of fp16 batch norm to fp32
    if self.dtype == dtypes.float16 or self.dtype == dtypes.bfloat16:
      return dtypes.float32
    else:
      return self.dtype or dtypes.float32

  def _support_zero_size_input(self):
    return distribution_strategy_context.has_strategy() and getattr(
        distribution_strategy_context.get_strategy().extended,
        'experimental_enable_get_next_as_optional', False)

  def build(self, input_shape):
    #tf.print('building')
    input_shape = tensor_shape.TensorShape(input_shape)
    if not input_shape.ndims:
      raise ValueError('Input has undefined rank:', input_shape)
    ndims = len(input_shape)

    # Convert axis to list and resolve negatives
    if isinstance(self.axis, int):
      self.axis = [self.axis]

    for idx, x in enumerate(self.axis):
      if x < 0:
        self.axis[idx] = ndims + x

    # Validate axes
    for x in self.axis:
      if x < 0 or x >= ndims:
        raise ValueError('Invalid axis: %d' % x)
    if len(self.axis) != len(set(self.axis)):
      raise ValueError('Duplicate axis: %s' % self.axis)

    axis_to_dim = {x: input_shape.dims[x].value for x in self.axis}
    for x in axis_to_dim:
      if axis_to_dim[x] is None:
        raise ValueError('Input has undefined `axis` dimension. Input shape: ',
                         input_shape)
    self.input_spec = InputSpec(ndim=ndims, axes=axis_to_dim)

    if len(axis_to_dim) == 1 and self.virtual_batch_size is None:
      # Single axis batch norm (most common/default use-case)
      param_shape = (list(axis_to_dim.values())[0],)
    else:
      # Parameter shape is the original shape but with 1 in all non-axis dims
      param_shape = [axis_to_dim[i] if i in axis_to_dim
                     else 1 for i in range(ndims)]
      if self.virtual_batch_size is not None:
        # When using virtual batches, add an extra dim at index 1
        param_shape.insert(1, 1)
        for idx, x in enumerate(self.axis):
          self.axis[idx] = x + 1      # Account for added dimension
    param_shape = (self.filtersize*self.filtersize*input_shape[-1])
    #tf.print(param_shape)
    if self.scale:
      self.gamma = self.add_weight(
          name='gamma',
          shape=param_shape,
          dtype=self._param_dtype,
          initializer=self.gamma_initializer,
          regularizer=self.gamma_regularizer,
          constraint=self.gamma_constraint,
          trainable=True,
          experimental_autocast=False)
    else:
      self.gamma = None
      if self.fused:
        self._gamma_const = K.constant(
            1.0, dtype=self._param_dtype, shape=param_shape)

    if self.center:
      self.beta = self.add_weight(
          name='beta',
          shape=param_shape,
          dtype=self._param_dtype,
          initializer=self.beta_initializer,
          regularizer=self.beta_regularizer,
          constraint=self.beta_constraint,
          trainable=True,
          experimental_autocast=False)
    else:
      self.beta = None
      if self.fused:
        self._beta_const = K.constant(
            0.0, dtype=self._param_dtype, shape=param_shape)

    try:
      # Disable variable partitioning when creating the moving mean and variance
      if hasattr(self, '_scope') and self._scope:
        partitioner = self._scope.partitioner
        self._scope.set_partitioner(None)
      else:
        partitioner = None
      self.moving_mean = self.add_weight(
          name='moving_mean',
          shape=param_shape,
          dtype=self._param_dtype,
          initializer=self.moving_mean_initializer,
          synchronization=tf_variables.VariableSynchronization.ON_READ,
          trainable=False,
          aggregation=tf_variables.VariableAggregation.MEAN,
          experimental_autocast=False)

      self.moving_variance = self.add_weight(
          name='moving_variance',
          shape=param_shape,
          dtype=self._param_dtype,
          initializer=self.moving_variance_initializer,
          synchronization=tf_variables.VariableSynchronization.ON_READ,
          trainable=False,
          aggregation=tf_variables.VariableAggregation.MEAN,
          experimental_autocast=False)

    finally:
      if partitioner:
        self._scope.set_partitioner(partitioner)
    self.built = True

  def _assign_moving_average(self, variable, value, momentum, inputs_size):
    with K.name_scope('AssignMovingAvg') as scope:
      with ops.colocate_with(variable):
        decay = ops.convert_to_tensor_v2(1.0 - momentum, name='decay')
        if decay.dtype != variable.dtype.base_dtype:
          decay = math_ops.cast(decay, variable.dtype.base_dtype)
        update_delta = (
            variable - math_ops.cast(value, variable.dtype)) * decay
        if inputs_size is not None:
          update_delta = array_ops.where(inputs_size > 0, update_delta,
                                         K.zeros_like(update_delta))
        return state_ops.assign_sub(variable, update_delta, name=scope)

  def _assign_new_value(self, variable, value):
    with K.name_scope('AssignNewValue') as scope:
      with ops.colocate_with(variable):
        return state_ops.assign(variable, value, name=scope)

 

    train_op = _fused_batch_norm_training
    if use_fused_avg_updates and input_batch_size is not None:
      # pylint: disable=g-long-lambda
      train_op = lambda: tf_utils.smart_cond(input_batch_size > 0,
                                             _fused_batch_norm_training,
                                             _fused_batch_norm_training_empty)
      # pylint: enable=g-long-lambda

    output, mean, variance = tf_utils.smart_cond(training, train_op,
                                                 _fused_batch_norm_inference)
    variance = _maybe_add_or_remove_bessels_correction(variance, remove=True)

    training_value = tf_utils.constant_value(training)
    if training_value or training_value is None:
      if not use_fused_avg_updates:
        if training_value is None:
          momentum = tf_utils.smart_cond(training, lambda: self.momentum,
                                         lambda: 1.0)
        else:
          momentum = ops.convert_to_tensor_v2(self.momentum)

      def mean_update():
        """Update self.moving_mean with the most recent data point."""
        if use_fused_avg_updates:
          return self._assign_new_value(self.moving_mean, mean)
        else:
          return self._assign_moving_average(self.moving_mean, mean, momentum,
                                             input_batch_size)

      def variance_update():
        """Update self.moving_variance with the most recent data point."""
        if use_fused_avg_updates:
          return self._assign_new_value(self.moving_variance, variance)
        else:
          return self._assign_moving_average(self.moving_variance, variance,
                                             momentum, input_batch_size)

      self.add_update(mean_update)
      self.add_update(variance_update)

    return output


  # def _calculate_mean_and_var(self, inputs, reduction_axes, keep_dims):
    # return nn.moments(inputs, reduction_axes, keep_dims=keep_dims)

  # def _moments(self, inputs, reduction_axes, keep_dims):
    # mean, variance = self._calculate_mean_and_var(inputs, reduction_axes,
                                                  # keep_dims)
    # # TODO(b/129279393): Support zero batch input in non DistributionStrategy
    # # code as well.
    # if self._support_zero_size_input():
      # input_batch_size = array_ops.shape(inputs)[0]
      # mean = array_ops.where(input_batch_size > 0, mean, K.zeros_like(mean))
      # variance = array_ops.where(input_batch_size > 0, variance,
                                 # K.zeros_like(variance))
    # return mean, variance
	
  def _calculate_vector_mean_and_var(self, inputs, reduction_axes, keep_dims):

    
    N = tf.shape(inputs)[0]
    C = tf.shape(inputs)[-1]
    patches = tf.image.extract_patches(inputs, [1,self.filtersize,self.filtersize,1], strides=[1,self.filtersize,self.filtersize,1], rates = [1,1,1,1], padding='VALID')
	
    patches = tf.reshape(patches, [-1,C*(self.filtersize**2)])
    mean, variance = tf.nn.moments(patches,0)
    #tf.print(mean)
    return mean, variance

  def _get_training_value(self, training=None):
    if training is None:
      training = K.learning_phase()
    if self._USE_V2_BEHAVIOR:
      if isinstance(training, int):
        training = bool(training)
      if base_layer_utils.is_in_keras_graph():
        training = math_ops.logical_and(training, self._get_trainable_var())
      elif not self.trainable:
        # When the layer is not trainable, it overrides the value passed from
        # model.
        training = self.trainable
    return training

  def call(self, inputs_nopad, training=None):
    
    N = tf.shape(inputs_nopad)[0]
    #H = tf.shape(inputs_nopad)[1]
    #N = inputs_nopad.shape[0]
    H = inputs_nopad.shape[1]
    remain = self.filtersize - tf.math.floormod(H,self.filtersize);
    #C = tf.shape(inputs_nopad)[-1]	
    C = inputs_nopad.shape[-1]
    right_side = tf.math.floordiv(remain,2)
    leftover = tf.math.floormod(remain,2)
    left_side = tf.add(right_side,leftover)
    

    
    pad_left = tf.zeros((N,left_side,H,C))
    pad_right = tf.zeros((N,right_side, H,C))
    pad_top = tf.zeros((N,tf.add(tf.add(H,left_side),right_side), left_side, C))
    pad_bottom = tf.zeros((N, tf.add(tf.add(H,left_side),right_side), right_side, C))
    inputs = tf.concat([pad_left, inputs_nopad], axis=1)
    inputs = tf.concat([inputs, pad_right], axis=1)
    inputs = tf.concat([pad_top, inputs], axis=2)
    inputs = tf.concat([inputs, pad_bottom], axis=2)
    #input_shape = tf.TensorShape([N,tf.add(H,left_side,right_side), tf.add(H,left_side,right_side), C])

    #tf.print('calling', output_stream=sys.stderr)
    training = self._get_training_value(training)

    # Compute the axes along which to reduce the mean / variance
    input_shape = inputs.shape
    ndims = len(input_shape)
    reduction_axes = [i for i in range(ndims) if i not in self.axis]

    # if self.virtual_batch_size is not None:
      # del reduction_axes[1]     # Do not reduce along virtual batch dim

    # Broadcasting only necessary for single-axis batch norm where the axis is
    # not the last dimension
    broadcast_shape = [1] * ndims
    broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value
    def _broadcast(v):
      if (v is not None and len(v.shape) != ndims and
          reduction_axes != list(range(ndims - 1))):
        return array_ops.reshape(v, broadcast_shape)
      return v

    scale, offset = _broadcast(self.gamma), _broadcast(self.beta)
    
    def _compose_transforms(scale, offset, then_scale, then_offset):
      if then_scale is not None:
        scale *= then_scale
        offset *= then_scale
      if then_offset is not None:
        offset += then_offset
      return (scale, offset)

    # Determine a boolean value for `training`: could be True, False, or None.
    training_value = tf_utils.constant_value(training)
    #tf.print('training value', training_value) 
	
    if training_value == False:  # pylint: disable=singleton-comparison,g-explicit-bool-comparison
      mean, variance = self.moving_mean, self.moving_variance
      #tf.print(mean, variance, summarize=-1)
      #tf.print('training value is False')
    else:
      if self.adjustment:
        adj_scale, adj_bias = self.adjustment(array_ops.shape(inputs))
        # Adjust only during training.
        adj_scale = tf_utils.smart_cond(training,
                                        lambda: adj_scale,
                                        lambda: array_ops.ones_like(adj_scale))
        adj_bias = tf_utils.smart_cond(training,
                                       lambda: adj_bias,
                                       lambda: array_ops.zeros_like(adj_bias))
        scale, offset = _compose_transforms(adj_scale, adj_bias, scale, offset)

      # Some of the computations here are not necessary when training==False
      # but not a constant. However, this makes the code simpler.
      keep_dims = self.virtual_batch_size is not None or len(self.axis) > 1
      #mean, variance = self._moments(
      #    math_ops.cast(inputs, self._param_dtype),
      #    reduction_axes,
      #    keep_dims=keep_dims)
       
      mean, variance = self._calculate_vector_mean_and_var(math_ops.cast(inputs, self._param_dtype), reduction_axes, keep_dims=keep_dims)
      moving_mean = self.moving_mean
      moving_variance = self.moving_variance

      mean = tf_utils.smart_cond(training, lambda: mean,
                                 lambda: ops.convert_to_tensor_v2(moving_mean))

      variance = tf_utils.smart_cond(
          training, lambda: variance,
          lambda: ops.convert_to_tensor_v2(moving_variance))

      new_mean, new_variance = mean, variance

      if self._support_zero_size_input():
        # Keras assumes that batch dimension is the first dimension for Batch
        # Normalization.
        input_batch_size = array_ops.shape(inputs)[0]
      else:
        input_batch_size = None

      def _do_update(var, value):
        """Compute the updates for mean and variance."""
        return self._assign_moving_average(var, value, self.momentum,
                                           input_batch_size)

      def mean_update():
        true_branch = lambda: _do_update(self.moving_mean, new_mean)
        false_branch = lambda: self.moving_mean
        return tf_utils.smart_cond(training, true_branch, false_branch)

      def variance_update():
        """Update the moving variance."""

        def true_branch_renorm():
          # We apply epsilon as part of the moving_stddev to mirror the training
          # code path.
          moving_stddev = _do_update(self.moving_stddev,
                                     math_ops.sqrt(new_variance + self.epsilon))
          return self._assign_new_value(
              self.moving_variance,
              # Apply relu in case floating point rounding causes it to go
              # negative.
              K.relu(moving_stddev * moving_stddev - self.epsilon))

        if self.renorm:
          true_branch = true_branch_renorm
        else:
          true_branch = lambda: _do_update(self.moving_variance, new_variance)

        false_branch = lambda: self.moving_variance
        return tf_utils.smart_cond(training, true_branch, false_branch)

      #if training, updates the moving mean with mean (in do_update), if testing, returns moving_mean
      self.add_update(mean_update)
      self.add_update(variance_update)

    mean = math_ops.cast(mean, inputs.dtype)
    variance = math_ops.cast(variance, inputs.dtype)
    if offset is not None:
      offset = math_ops.cast(offset, inputs.dtype)
    if scale is not None:
      scale = math_ops.cast(scale, inputs.dtype)  

    def _do_batchnorm(inputs, mean, variance, offset, scale, variance_epsilon, name=None):
      with ops.name_scope(name, "batchnorm", [inputs, mean, variance, scale, offset]):
        inv = math_ops.rsqrt(variance + variance_epsilon)
      if scale is not None:
        inv *= scale
      
      N = tf.shape(inputs)[0]
      H = tf.shape(inputs)[1]
      C = tf.shape(inputs)[-1]
      dilated_input = inputs
      dilated_h = H
      #dilated_h = tf.math.multiply(self.filtersize,tf.math.floordiv(H,self.filtersize))
      #tf.print(dilated_h)
      #dilated_input = inputs[:,:dilated_h,:dilated_h,:]
      patches = tf.image.extract_patches(dilated_input, [1,self.filtersize,self.filtersize,1], strides=[1,self.filtersize,self.filtersize,1], rates = [1,1,1,1], padding='VALID')
      patches = tf.reshape(patches, [-1,tf.math.multiply(tf.math.square(self.filtersize),C)])
      #tf.print(tf.shape(patches))
      #tf.print(tf.shape(mean))
      #tf.print(tf.shape(inv))
      patches_standardized = tf.add(tf.subtract(tf.math.multiply(patches,inv),tf.math.multiply(mean,inv)) , offset)
      axes_1_2_size = tf.cast(tf.sqrt(tf.math.divide(tf.math.square(dilated_h),tf.math.square(self.filtersize))), tf.int32)
      reconstruct = tf.reshape(patches_standardized, (N, axes_1_2_size, axes_1_2_size, self.filtersize, self.filtersize, C)) 
      reconstruct = tf.transpose(reconstruct, (0, 1, 3, 2, 4, 5))
      # Reshape back
      reconstruct = tf.reshape(reconstruct, (N, dilated_h, dilated_h, C))
      #tf.print(mean, variance)
      #reconstruct = tf.concat([reconstruct, inputs[:,dilated_h:H, :dilated_h, :]], axis=1)
      #reconstruct = tf.concat([reconstruct, inputs[:,:,dilated_h:H, :]], axis=2)
      #tf.print('appliedbnorm') #reconstruct)
      return reconstruct
	  

    outputs = _do_batchnorm(inputs, mean, variance, offset, scale, self.epsilon)

    #tf.print(tf.shape(outputs))
        # # Note: tensorflow/contrib/quantize/python/fold_batch_norms.py depends on
        # # the precise order of ops that are generated by the expression below.
      # return x * math_ops.cast(inv, x.dtype) + math_ops.cast(
        # offset - mean * inv if offset is not None else -mean * inv, x.dtype)
		
		
    # If some components of the shape got lost due to adjustments, fix that.
    outputs.set_shape(input_shape)

    if self.virtual_batch_size is not None:
      outputs = undo_virtual_batching(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    #tf.print('reached here')
    N = input_shape[0]
    H = input_shape[1]
    remain = self.filtersize - tf.math.floormod(H,self.filtersize);
    C = input_shape[-1]	
    each_side = tf.math.floordiv(remain,2)
    leftover = tf.math.floormod(remain,2)
    left_side = tf.add(each_side,leftover)
    right_side = each_side
    out_shape = tf.TensorShape([N,H, H+left_side+right_side, C])

    return out_shape

  def get_config(self):
    config = {
        'axis': self.axis,
        'momentum': self.momentum,
        'epsilon': self.epsilon,
        'center': self.center,
        'scale': self.scale,
        'beta_initializer': initializers.serialize(self.beta_initializer),
        'gamma_initializer': initializers.serialize(self.gamma_initializer),
        'moving_mean_initializer':
            initializers.serialize(self.moving_mean_initializer),
        'moving_variance_initializer':
            initializers.serialize(self.moving_variance_initializer),
        'beta_regularizer': regularizers.serialize(self.beta_regularizer),
        'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
        'beta_constraint': constraints.serialize(self.beta_constraint),
        'gamma_constraint': constraints.serialize(self.gamma_constraint)
    }
    # Only add TensorFlow-specific parameters if they are set, so as to preserve
    # model compatibility with external Keras.
    if self.renorm:
      config['renorm'] = True
      config['renorm_clipping'] = self.renorm_clipping
      config['renorm_momentum'] = self.renorm_momentum
    if self.virtual_batch_size is not None:
      config['virtual_batch_size'] = self.virtual_batch_size
    # Note: adjustment is not serializable.
    if self.adjustment is not None:
      logging.warning('The `adjustment` function of this `BatchNormalization` '
                      'layer cannot be serialized and has been omitted from '
                      'the layer config. It will not be included when '
                      're-creating the layer from the saved config.')
    base_config = super(CBatchNormalizationBase, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def replace_in_base_docstring(replacements):
  string = CBatchNormalizationBase.__doc__
  for old, new in replacements:
    assert old in string
    string = string.replace(old, new)
  return string


@keras_export(v1=['keras.layers.BatchNormalization'])  # pylint: disable=missing-docstring
class CustomBatchNormalization(CBatchNormalizationBase):

  __doc__ = replace_in_base_docstring(
      [('''
    fused: if `True`, use a faster, fused implementation, or raise a ValueError
      if the fused implementation cannot be used. If `None`, use the faster
      implementation if possible. If False, do not used the fused
      implementation.''',
        '''
    fused: if `None` or `True`, use a faster, fused implementation if possible.
      If `False`, use the system recommended implementation.'''),
       ('{{TRAINABLE_ATTRIBUTE_NOTE}}', '')])

  _USE_V2_BEHAVIOR = False

