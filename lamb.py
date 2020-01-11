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
# =============================================================================

# pylint: disable=unused-import,g-bad-import-order

"""Lamb for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.platform import tf_logging as logging


@keras_export('keras.optimizers.Lamb')
class Lamb(optimizer_v2.OptimizerV2):
  """Optimizer that implements the Lamb algorithm.

  Default parameters follow those provided in the original paper.
  # Arguments
      lr: float >= 0. Learning rate.
      beta_1: float, 0 < beta < 1. Generally close to 1.
      beta_2: float, 0 < beta < 1. Generally close to 1.
      epsilon: float >= 0. Fuzz factor. If `None`, defaults to 1e-6.
      weight_decay: float >= 0. Weight decay regularization.
      decay: float >= 0. Learning rate decay over each update.
  # References
    - [Reducing BERT Pre-Training Time from 3 Days to 76 Minutes]
        (https://arxiv.org/abs/1904.00962)
    - https://towardsdatascience.com/an-intuitive-understanding-of-the-lamb-optimizer-46f8c0ae4866
  """
  
  def __init__(self,
               learning_rate=1e-3,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               weight_decay=0.0001,
               accum_iters=1,
               name='Lamb',
               **kwargs):
    if not 0.0 <= beta_1 < 1.0:
        raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
    if not 0.0 <= beta_2 < 1.0:
        raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
    super(Lamb, self).__init__(name, **kwargs)
    
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('weight_decay', kwargs.get('weight_decay', weight_decay))
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self._set_hyper('accum_iters', accum_iters)
    self.epsilon = epsilon or backend_config.epsilon()
  
  def _create_slots(self, var_list):
    # Create slots for the first and second moments.
    # Separate for-loops to respect the ordering of slot variables from v1.
    for var in var_list:
      self.add_slot(var, 'm')
    for var in var_list:
      self.add_slot(var, 'v')
  
  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(Lamb, self)._prepare_local(var_device, var_dtype, apply_state)
    
    local_step      = math_ops.cast(self.iterations + 1, var_dtype)
    beta_1_t        = array_ops.identity(self._get_hyper('beta_1', var_dtype))
    beta_2_t        = array_ops.identity(self._get_hyper('beta_2', var_dtype))
    beta_1_power    = math_ops.pow(beta_1_t, local_step)
    beta_2_power    = math_ops.pow(beta_2_t, local_step)
    lr_t            = array_ops.identity(self._get_hyper('learning_rate', var_dtype))
    weight_decay    = array_ops.identity(self._get_hyper('weight_decay', var_dtype))
    apply_state[(var_device, var_dtype)].update(dict(
        lr=lr_t,
        epsilon=ops.convert_to_tensor(self.epsilon, var_dtype),
        beta_1_t=beta_1_t,
        beta_1_power=beta_1_power,
        one_minus_beta_1_t=1 - beta_1_t,
        one_minus_beta_1_power=1 - beta_1_power,
        beta_2_t=beta_2_t,
        beta_2_power=beta_2_power,
        one_minus_beta_2_t=1 - beta_2_t,
        one_minus_beta_2_power=1 - beta_2_power,
        weight_decay=weight_decay
    ))
  
  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))
    
    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')
    
    m_t = coefficients['beta_1_t'] * m + coefficients['one_minus_beta_1_t'] * grad
    m_t = m_t / coefficients['one_minus_beta_1_power']
    m_t = state_ops.assign(m, m_t, use_locking=self._use_locking)
    
    v_t = coefficients['beta_2_t'] * v + coefficients['one_minus_beta_2_t'] * (grad * grad)
    v_t = v_t / coefficients['one_minus_beta_2_power']
    v_t = state_ops.assign(v, v_t, use_locking=self._use_locking)

    r_t = m_t / (math_ops.sqrt(v_t) + coefficients['epsilon'])
    r_t = r_t + (coefficients['weight_decay'] * var)
    
    numerator   = linalg_ops.norm(var, ord=2)
    denominator = linalg_ops.norm(r_t, ord=2)
    trust_ratio = array_ops.where(
        math_ops.greater(numerator, 0),
        array_ops.where(
            math_ops.greater(denominator, 0),
            (numerator / denominator),
            1.0),
        1.0)
    
    return state_ops.assign_sub(
                    var, 
                    (coefficients['lr'] * trust_ratio) * r_t , 
                    use_locking=self._use_locking).op
  
  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')
    
    m_t = state_ops.assign(m, m * coefficients['beta_1_t'], use_locking=self._use_locking)
    with ops.control_dependencies([m_t]):
      m_t = self._resource_scatter_add(m, indices, grad * coefficients['one_minus_beta_1_t'])
    m_t = state_ops.assign(m, m_t / coefficients['one_minus_beta_1_power'], use_locking=self._use_locking)
    
    v_t = state_ops.assign(v, v * coefficients['beta_2_t'], use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
      v_t = self._resource_scatter_add(v, indices, (grad * grad) * coefficients['one_minus_beta_2_t'])
    v_t = state_ops.assign(v, v_t / coefficients['one_minus_beta_2_power'], use_locking=self._use_locking)

    r_t = m_t / (math_ops.sqrt(v_t) + coefficients['epsilon'])
    r_t = r_t + (coefficients['weight_decay'] * var)
    
    numerator   = linalg_ops.norm(var, ord=2)
    denominator = linalg_ops.norm(r_t, ord=2)
    trust_ratio = array_ops.where(
        math_ops.greater(numerator, 0),
        array_ops.where(
            math_ops.greater(denominator, 0),
            (numerator / denominator),
            1.0),
        1.0)
    
    var_t = state_ops.assign_sub(
                    var, 
                    (coefficients['lr'] * trust_ratio) * r_t , 
                    use_locking=self._use_locking)
    return control_flow_ops.group(*[var_t, m_t, v_t])
  
  def get_config(self):
    config = super(Lamb, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._serialize_hyperparameter('decay'),
        'beta_1': self._serialize_hyperparameter('beta_1'),
        'beta_2': self._serialize_hyperparameter('beta_2'),
        'epsilon': self.epsilon
    })
    return config
