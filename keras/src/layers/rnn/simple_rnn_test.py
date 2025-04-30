import numpy as np
import pytest

from keras.src import initializers
from keras.src import layers
from keras.src import testing


class SimpleRNNTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_basics(self):
        self.run_layer_test(
            layers.SimpleRNN,
            init_kwargs={"units": 3, "dropout": 0.5, "recurrent_dropout": 0.5},
            input_shape=(3, 2, 4),
            call_kwargs={"training": True},
            expected_output_shape=(3, 3),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            expected_num_non_trainable_variables=1,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.SimpleRNN,
            init_kwargs={
                "units": 3,
                "return_sequences": True,
                "bias_regularizer": "l1",
                "kernel_regularizer": "l2",
                "recurrent_regularizer": "l2",
            },
            input_shape=(3, 2, 4),
            expected_output_shape=(3, 2, 3),
            expected_num_losses=3,
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            supports_masking=True,
        )

    def test_correctness(self):
        sequence = np.arange(24).reshape((2, 3, 4)).astype("float32")
        layer = layers.SimpleRNN(
            4,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.405432, 0.405432, 0.405432, 0.405432],
                    [0.73605347, 0.73605347, 0.73605347, 0.73605347],
                ]
            ),
            output,
        )
        layer = layers.SimpleRNN(
            4,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            unroll=True,
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.405432, 0.405432, 0.405432, 0.405432],
                    [0.73605347, 0.73605347, 0.73605347, 0.73605347],
                ]
            ),
            output,
        )

        layer = layers.SimpleRNN(
            4,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            go_backwards=True,
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.11144729, 0.11144729, 0.11144729, 0.11144729],
                    [0.5528889, 0.5528889, 0.5528889, 0.5528889],
                ]
            ),
            output,
        )
        layer = layers.SimpleRNN(
            4,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            go_backwards=True,
            unroll=True,
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.11144729, 0.11144729, 0.11144729, 0.11144729],
                    [0.5528889, 0.5528889, 0.5528889, 0.5528889],
                ]
            ),
            output,
        )

    def test_statefulness(self):
        sequence = np.arange(24).reshape((2, 3, 4)).astype("float32")
        layer = layers.SimpleRNN(
            4,
            stateful=True,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
        )
        layer(sequence)
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.40559256, 0.40559256, 0.40559256, 0.40559256],
                    [0.7361247, 0.7361247, 0.7361247, 0.7361247],
                ]
            ),
            output,
        )
        layer.reset_state()
        layer(sequence)
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.40559256, 0.40559256, 0.40559256, 0.40559256],
                    [0.7361247, 0.7361247, 0.7361247, 0.7361247],
                ]
            ),
            output,
        )

    def test_pass_initial_state(self):
        sequence = np.arange(24).reshape((2, 4, 3)).astype("float32")
        initial_state = np.arange(8).reshape((2, 4)).astype("float32")
        layer = layers.SimpleRNN(
            4,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
        )
        output = layer(sequence, initial_state=initial_state)
        self.assertAllClose(
            np.array(
                [
                    [0.33621645, 0.33621645, 0.33621645, 0.33621645],
                    [0.6262637, 0.6262637, 0.6262637, 0.6262637],
                ]
            ),
            output,
        )

        layer = layers.SimpleRNN(
            4,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            go_backwards=True,
        )
        output = layer(sequence, initial_state=initial_state)
        self.assertAllClose(
            np.array(
                [
                    [0.07344437, 0.07344437, 0.07344437, 0.07344437],
                    [0.43043602, 0.43043602, 0.43043602, 0.43043602],
                ]
            ),
            output,
        )

    def test_masking(self):
        sequence = np.arange(24).reshape((2, 4, 3)).astype("float32")
        mask = np.array([[True, True, False, True], [True, False, False, True]])
        layer = layers.SimpleRNN(
            4,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            unroll=True,
        )
        output = layer(sequence, mask=mask)
        self.assertAllClose(
            np.array(
                [
                    [0.32951632, 0.32951632, 0.32951632, 0.32951632],
                    [0.61799484, 0.61799484, 0.61799484, 0.61799484],
                ]
            ),
            output,
        )

        layer = layers.SimpleRNN(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            return_sequences=True,
        )
        output = layer(sequence, mask=mask)
        self.assertAllClose(
            np.array(
                [
                    [0.0599281, 0.0599281],
                    [0.15122814, 0.15122814],
                    [0.15122814, 0.15122814],
                    [0.32394567, 0.32394567],
                ],
            ),
            output[0],
        )
        self.assertAllClose(
            np.array(
                [
                    [0.3969304, 0.3969304],
                    [0.3969304, 0.3969304],
                    [0.3969304, 0.3969304],
                    [0.608085, 0.608085],
                ],
            ),
            output[1],
        )

        layer = layers.SimpleRNN(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            return_sequences=True,
            zero_output_for_mask=True,
        )
        output = layer(sequence, mask=mask)
        self.assertAllClose(
            np.array(
                [
                    [0.0599281, 0.0599281],
                    [0.15122814, 0.15122814],
                    [0.0, 0.0],
                    [0.32394567, 0.32394567],
                ],
            ),
            output[0],
        )
        self.assertAllClose(
            np.array(
                [
                    [0.3969304, 0.3969304],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.608085, 0.608085],
                ],
            ),
            output[1],
        )

        layer = layers.SimpleRNN(
            4,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            go_backwards=True,
        )
        output = layer(sequence, mask=mask)
        self.assertAllClose(
            np.array(
                [
                    [0.07376196, 0.07376196, 0.07376196, 0.07376196],
                    [0.43645123, 0.43645123, 0.43645123, 0.43645123],
                ]
            ),
            output,
        )


import numpy as np
import tensorflow as tf
from keras.src.backend.tensorflow.rnn import rnn


def test_rnn_time_major():
    batch_size, timesteps, input_dim, units = 2, 3, 4, 5

    inputs = tf.random.normal((timesteps, batch_size, input_dim)) 
    initial_state = tf.zeros((batch_size, units))

    W = tf.ones((input_dim, units))
    U = tf.eye(units)

    def step_fn(input_t, states):
        prev_state = states[0]
        output = tf.matmul(input_t, W) + tf.matmul(prev_state, U)
        return output, [output]

    final_state, outputs, all_states = rnn(
        step_fn,
        inputs,
        [initial_state],
        go_backwards=False,
        unroll=False,
        input_length=timesteps,
        time_major=True,
        zero_output_for_mask=False,
        return_all_outputs=True,
    )

    assert outputs.shape == (timesteps, batch_size, units) 


def test_rnn_with_constants():
    batch_size, timesteps, input_dim, units = 2, 3, 4, 5

    inputs = tf.random.normal((batch_size, timesteps, input_dim))
    initial_state = tf.zeros((batch_size, units))
    constant_value = tf.ones((batch_size, units))

    def step_fn(input_t, states):
        prev_state = states[0]
        constant = states[1]  # get constant
        output = tf.matmul(input_t, tf.ones((input_dim, units))) + tf.matmul(prev_state + constant, tf.eye(units))
        return output, [output]

    final_state, outputs, all_states = rnn(
        step_fn,
        inputs,
        [initial_state],
        constants=[constant_value],
        go_backwards=False,
        unroll=False,
        input_length=timesteps,
        zero_output_for_mask=False,
        return_all_outputs=True,
    )

    assert outputs.shape == (batch_size, timesteps, units)



def test_expand_mask_high_rank_diff():
    batch_size, timesteps, input_dim, units = 2, 3, 4, 5
    inputs = tf.random.normal((batch_size, timesteps, input_dim, 2))
    mask = tf.ones((batch_size, timesteps), dtype=tf.int32) 
    mask = tf.cast(mask, tf.bool)

    initial_state = [tf.zeros((batch_size, units))]

    def step_fn(input_t, states):
        return tf.reduce_mean(input_t, axis=-1), states

    rnn(
        step_fn,
        inputs,
        initial_state,
        mask=mask,
        unroll=True,
        input_length=timesteps
    )


def test_rnn_return_last_output_only():
    batch_size, timesteps, input_dim, units = 2, 3, 4, 5

    inputs = tf.random.normal((batch_size, timesteps, input_dim))
    initial_state = tf.zeros((batch_size, units))

    W = tf.ones((input_dim, units))
    U = tf.eye(units)

    def step_fn(input_t, states):
        return tf.matmul(input_t, W) + tf.matmul(states[0], U), [states[0]]

    final_state, outputs, new_states = rnn(
        step_fn,
        inputs,
        [initial_state],
        return_all_outputs=False,
        unroll=True
    )

    assert outputs.shape == (batch_size, 1, units)

