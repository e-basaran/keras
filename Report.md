# Report for assignment 3


## Project

Name: Keras

URL: https://github.com/e-basaran/keras.git

Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow. It provides a simple interface for building neural networks using the building blocks of neural networks.

## Onboarding experience

For our group members using macOS there was a problem with the installation of tensorflow and torch. "pip install -r requirements.txt" did not work. We had to install them separetely. The issue was mainly about the version conflicts between the dependencies. We were first using Python3.13, then realized that it was not compatible with the tensorflow 2.18.0, so we changed to Python3.9, and explicitly stated Python3.9 in each installation command. That fixed most of our onboarding issues. 


## Complexity

Running the lizard tool on the Keras codebase showed these results about its complexity:

1. Overall Statistics:
- Total lines of code (NLOC): 173,365
- Average NLOC per function: 12.5
- Average Cyclomatic Complexity Number (CCN): 2.2
- Average token count per function: 91.3
- Total function count: 11,302
- Number of functions with high complexity: 120 (1% of total functions)

Among these 120 functions, we selected the following four functions (as we had only 4 members):
- `deserialize_keras_object` in `/keras/src/saving/serialization_lib.py` (CCN: 53, NLOC: 212, Parameters: 4)
- `rnn` in `keras/src/backend/tensorflow/rnn.py` (CCN: 39, NLOC: 202, Parameters: 11)
- `set_vocabulary` in `keras/keras/src/layers/preprocessing/index_lookup.py` (CCN: 31, NLOC: 139, Parameters: 3)
- `update_confusion_matrix_variables` in `keras/keras/src/metrics/metrics_utils.py` (CCN: 34, NLOC: 161, Parameters: 10)

The complexity values given above are from the lizard analysis. We also manually counted CCN values for these functions. 

- `deserialize_keras_object` has 53 CCN, and we counted 53 decision points manually, which would give 53 + 1 = 54 CCN because of the CC = 1 + D rule.
- `update_confusion_matrix_variables` has 34 CCN, and we counted 36.
- `rnn` has 39 CCN, and we counted 39.
- `set_vocabulary` has 31 CCN, and we counted 30 decision points manually, which would give 30 + 1 = 31 CCN.

The results are not exactly the same, but they are close. The deviataions might be due to the different ways of counting the decision points like for `if a and b` decision points or `try/except` blocks. In our measurements, we took exceptions into account. 

Upon analysis, we saw that functions were both complex and long. In our case, NLOC increased with the CCN. 

The purpose of the functions are as follows:

1. deserialize_keras_object in /keras/src/saving/serialization_lib.py:
This function retrieves and reconstructs Keras objects (like layers, optimizers, metrics) from their serialized configuration dictionaries. It handles complex deserialization logic including custom objects, nested configurations, and special cases like tensors and numpy arrays while enforcing safety checks and proper object reconstruction.

2. rnn in keras/src/backend/tensorflow/rnn.py:
This function implements the core recurrent neural network (RNN) computation by iterating over the time dimension of an input tensor and applying a step function at each timestep. It handles various RNN configurations including masking, bidirectional processing, and different input formats while managing state transitions and output collection.

3. set_vocabulary in keras/keras/src/layers/preprocessing/index_lookup.py:
This function sets up a vocabulary (and optionally document frequency weights) for text processing layers by directly specifying the vocabulary terms instead of learning them through adaptation. It performs extensive validation of the vocabulary, handles special tokens (like OOV and mask tokens), and ensures proper configuration for different output modes like TF-IDF.

4. update_confusion_matrix_variables in keras/keras/src/metrics/metrics_utils.py:
This function updates confusion matrix variables (true positives, false positives, true negatives, false negatives) based on predictions and ground truth labels with support for multiple thresholds. It handles various cases including multi-label classification, sample weights, and class-specific evaluations while providing optimizations for evenly distributed thresholds.

Overall, it can be said that the documentation is somewhat clear, but it is still not very detailed. 


## Refactoring

Refactoring plan for `deserialize_keras_object` (Eyüp Ahmet Başaran):
```python
def deserialize_keras_object(config, custom_objects=None, safe_mode=True, **kwargs):
     """Main deserialization function with reduced complexity through helper functions."""
     if not _is_valid_config(config):
         return _handle_invalid_config(config)
     
     custom_objects = _prepare_custom_objects(custom_objects, kwargs)
     
     if _is_special_type(config):
         return _deserialize_special_type(config)
         
     return _deserialize_standard_object(config, custom_objects, safe_mode)

def _is_valid_config(config):
     """Validates basic config structure."""
     return config is not None and isinstance(config, (dict, str, list, tuple))

def _handle_invalid_config(config):
     """Handles None or invalid config cases."""
     if config is None:
         return None
     if isinstance(config, PLAIN_TYPES):
         return config
     # ... other invalid cases

def _prepare_custom_objects(custom_objects, kwargs):
     """Merges and prepares custom objects dictionary."""
     # ... custom objects preparation logic

def _is_special_type(config):
     """Checks if config represents a special type (tensor, numpy array, etc)."""
     # ... special type checking logic

def _deserialize_special_type(config):
     """Handles deserialization of special types."""
     # ... special type deserialization

def _deserialize_standard_object(config, custom_objects, safe_mode):
     """Handles standard Keras object deserialization."""
     # ... main deserialization logic
```

Positive effects would be that the code would be more readable and testable, also low CCN values are better for maintainability. But on the other hand, it would require more lines overall, and there would be a slight performance overhead from additional function calls.

After carrying out the refactoring, the CCN values of the functions decreased 70%, and the code became more readable and testable.

git diff master refactor/deserialize_keras_object

Refactoring plan for `set_vocabulary` (Bingjie Zhao):

The set_vocabulary function originally had high cyclomatic complexity due to multiple conditionals and logic checks. To reduce this complexity, we refactored the function by breaking it into smaller, more focused helper functions.
We split the original set_vocabulary function into several smaller functions, each focused on a specific aspect of the vocabulary setting process. Each smaller function only has 2-6 CC, which reduced > 35%. The new functions are:
_validate_idf_weights: Validates if the idf_weights parameter is correctly set when output_mode is "tf_idf".
_process_vocabulary: Handles vocabulary preprocessing, including checking if it's a file path or a tensor and converting it into a NumPy array if necessary.
_check_for_empty_vocabulary: Checks if the vocabulary is empty.
_check_for_repeated_tokens: Ensures that there are no repeated tokens in the vocabulary.
_check_special_tokens: Checks that special tokens (like mask and OOV tokens) are in the correct place.
_get_tokens: Extracts the actual tokens from the vocabulary, considering special tokens.
_check_vocabulary_size: Verifies if the vocabulary size is within the allowed limit (max_tokens).
_process_idf_weights: Handles the processing of idf_weights, ensuring they match the vocabulary size and are of the correct format.
_pad_idf_weights: Pads idf_weights to align with the vocabulary size, if necessary.
Code:
```python
def set_vocabulary(self, vocabulary, idf_weights=None):  
    """Sets vocabulary (and optionally document frequency) for this layer."""  
      
    self._validate_idf_weights(idf_weights)  
    vocabulary = self._process_vocabulary(vocabulary)  
      
    self._check_for_empty_vocabulary(vocabulary)  
    self._check_for_repeated_tokens(vocabulary)  
    self._check_special_tokens(vocabulary)  
      
    tokens = self._get_tokens(vocabulary)  
      
    self._check_vocabulary_size(tokens)  
    self.lookup_table = self._lookup_table_from_tokens(tokens)  
    self._record_vocabulary_size()  
  
    if self.output_mode == "tf_idf" and idf_weights is not None:  
        self._process_idf_weights(idf_weights, len(vocabulary))  
  
def _validate_idf_weights(self, idf_weights):  
    """Validates the idf_weights parameter."""  
    if self.output_mode == "tf_idf":  
        if idf_weights is None:  
            raise ValueError("`idf_weights` must be set if output_mode is 'tf_idf'.")  
    elif idf_weights is not None:  
        raise ValueError("`idf_weights` should only be set if output_mode is 'tf_idf'.")  
  
def _process_vocabulary(self, vocabulary):  
    """Processes vocabulary, converts to numpy array if needed."""  
    if isinstance(vocabulary, str):  
        self._check_vocabulary_file(vocabulary)  
        return self._lookup_table_from_file(vocabulary)  
    if tf.is_tensor(vocabulary):  
        return self._tensor_vocab_to_numpy(vocabulary)  
    elif isinstance(vocabulary, (list, tuple)):  
        return np.array(vocabulary)  
    return vocabulary  
  
def _check_vocabulary_file(self, vocabulary_file):  
    """Checks if the vocabulary file exists and is valid."""  
    if not tf.io.gfile.exists(vocabulary_file):  
        raise ValueError(f"Vocabulary file {vocabulary_file} does not exist.")  
    if self.output_mode == "tf_idf":  
        raise ValueError("output_mode 'tf_idf' does not support loading a vocabulary from file.")  
  
def _check_for_empty_vocabulary(self, vocabulary):  
    """Checks if the vocabulary is empty."""  
    if vocabulary.size == 0:  
        raise ValueError(f"Cannot set an empty vocabulary. Received: vocabulary={vocabulary}")  
  
def _check_for_repeated_tokens(self, vocabulary):  
    """Checks for repeated tokens in the vocabulary."""  
    repeated_tokens = self._find_repeated_tokens(vocabulary)  
    if repeated_tokens:  
        raise ValueError(f"The passed vocabulary has at least one repeated term. Please uniquify your dataset. The repeated terms are: {repeated_tokens}")  
  
def _check_special_tokens(self, vocabulary):  
    """Checks if special tokens are in the correct position."""  
    special_tokens = [self.mask_token] * self._oov_start_index() + [self.oov_token] * self.num_oov_indices  
    if np.array_equal(special_tokens, vocabulary[:self._token_start_index()]):  
        tokens = vocabulary[self._token_start_index():]  
    else:  
        tokens = vocabulary  
      
    if self.mask_token is not None and self.mask_token in tokens:  
        mask_index = np.argwhere(vocabulary == self.mask_token)[-1]  
        raise ValueError(f"Found reserved mask token at unexpected location in `vocabulary`. Received: mask_token={self.mask_token} at vocabulary index {mask_index}")  
  
    if self.oov_token is not None and self.invert and self.oov_token in tokens:  
        oov_index = np.argwhere(vocabulary == self.oov_token)[-1]  
        raise ValueError(f"Found reserved OOV token at unexpected location in `vocabulary`. Received: oov_token={self.oov_token} at vocabulary index {oov_index}")  
  
def _get_tokens(self, vocabulary):  
    """Extracts tokens from vocabulary after handling special tokens."""  
    oov_start = self._oov_start_index()  
    token_start = self._token_start_index()  
    special_tokens = [self.mask_token] * oov_start + [self.oov_token] * self.num_oov_indices  
    found_special_tokens = np.array_equal(special_tokens, vocabulary[:token_start])  
      
    return vocabulary[token_start:] if found_special_tokens else vocabulary  
  
def _check_vocabulary_size(self, tokens):  
    """Checks if the vocabulary size exceeds the maximum allowed size."""  
    new_vocab_size = self._token_start_index() + len(tokens)  
    if self.max_tokens is not None and new_vocab_size > self.max_tokens:  
        raise ValueError(f"Attempted to set a vocabulary larger than the maximum vocab size. Received vocabulary size is {new_vocab_size}; `max_tokens` is {self.max_tokens}.")  
  
def _process_idf_weights(self, idf_weights, vocab_length):  
    """Processes and validates the idf_weights."""  
    if len(idf_weights) != vocab_length:  
        raise ValueError(f"`idf_weights` must be the same length as vocabulary. len(idf_weights) is {len(idf_weights)}; len(vocabulary) is {vocab_length}")  
    idf_weights = self._convert_to_ndarray(idf_weights)  
    if idf_weights.ndim != 1:  
        raise ValueError(f"TF-IDF data must be a 1-index array. Received: type(idf_weights)={type(idf_weights)}")  
      
    # Padding of idf_weights if necessary  
    self._pad_idf_weights(idf_weights)  
  
def _pad_idf_weights(self, idf_weights):  
    """Pads the idf_weights if the vocabulary has no special tokens."""  
    front_padding_value = 0  
    if not np.array_equal([self.mask_token] * self._oov_start_index() + [self.oov_token] * self.num_oov_indices, self._get_tokens(idf_weights)[:self._token_start_index()]):  
        front_padding_value = np.average(idf_weights)  
      
    back_padding_value = 0  
    if self.pad_to_max_tokens and self.max_tokens is not None:  
        back_padding = self.max_tokens - len(idf_weights)  
    else:  
        back_padding = 0  
      
    weights = np.pad(idf_weights, (self._oov_start_index(), back_padding), "constant", constant_values=(front_padding_value, back_padding_value))  
    self.idf_weights = tf.Variable(weights, trainable=False)  
    self.idf_weights_const = self.idf_weights.value()  
```

Refactoring plan for `update_confusion_matrix_variables` (Melissa Saber):

To improve the update_confusion_matrix_variables function, the strategy is to break down its complex logic into smaller, focused helper functions. The main function will act as a coordinator, delegating specific tasks to these helper functions. This modular approach will not only reduce complexity but also enhance the overall quality, maintainability, and testability of the code.  The first step is to isolate all input validation and error handling into a dedicated function. This function will ensure that inputs are correctly formatted, check for incompatible parameters (such as label_weights with multi-label data), and validate the provided keys for updating the confusion matrix. Centralizing validation in this way will allow the main function to operate under the assumption that inputs are valid, enabling it to focus solely on executing its core logic.  Next, data preparation and type conversion will be handled by a separate function. This helper function will standardize input data by converting y_true and y_pred to the appropriate data types and reshaping sample_weight as needed. By managing all preprocessing tasks in one place, the main function can avoid repetitive type casting and reshaping, resulting in cleaner and more efficient code.  Handling thresholds, particularly in multi-label and top-k scenarios, introduces additional complexity. To address this, a dedicated function will manage threshold processing, including tiling and reshaping threshold arrays and accommodating special cases such as evenly distributed thresholds. Isolating this logic will help maintain a clear and concise main function, free from the clutter of threshold-specific branching.  Managing sample weights and label weights also requires careful handling of various scenarios. By creating a separate function for this task, we can streamline the application of sample weights, broadcasting and tiling them as needed, and incorporating label weights where applicable. This modularization will simplify the main function’s flow and allow for independent testing of the weight-handling logic.  Finally, the actual updating of the confusion matrix variables will be abstracted into a specialized function. This function will iterate over the required matrix conditions—such as true positives and false negatives—and apply the appropriate updates to the confusion matrix variables. By concentrating solely on updating logic, this function will keep the main function focused on high-level orchestration while ensuring that the matrix updates remain efficient and maintainable.  
Estimated impact of refactoring (lower CC, but other drawbacks?). 
- Reduced Cyclomatic Complexity: Splitting the function into smaller units is expected to bring the complexity down. 
- Improved Code Quality: Smaller functions which would be easier to maintain. 
- Enhanced Testability: Each helper function can be unit tested individually, leading to more granular and reliable tests.





## Coverage

### Tools

We used the `coverage.py` tool, which is the standard code coverage tool for Python. The experience was generally positive for several reasons:

1. Installation was straightforward using pip:
```bash
pip install coverage
```

2. The tool is well-documented with clear examples on its official documentation site (https://coverage.readthedocs.io/). The documentation covers everything from basic usage to advanced configuration options.

3. Integration with our build environment was simple.
   - Command line: `coverage run -m pytest` followed by `coverage report`

One challenge we encountered was configuring the tool to ignore certain paths and files that weren't relevant to our testing scope. 

### Your own coverage tool

git diff branch-coverage master

Our coverage tool supports:

   - if/else statements
   - Nested conditional statements
   - Providing binary (True/False) information for each branch execution
   - Accurately tracking which specific branches were executed

### Evaluation

   - Binary (True/False) tracking for each branch
   - Only tracks branch coverage, not line coverage
   - Requires manual instrumentation
   - Cannot track complex boolean expressions
   - No automatic reporting features
   - Limited to explicitly marked branches
   - Similar branch coverage patterns to coverage.py
   - Less comprehensive overall
   - Main difference: requires explicit instrumentation vs automatic tracking

## Coverage improvement


For deserialize_keras_object:

6 new test cases were added. Improved the coverage from 53% to 77%

For update_confusion_matrix_variables:

4 new test cases were added. Improved the coverage from 8% to 9%.

For set_vocabulary:

5 new test cases were added. Improved the coverage from 32% to 51%.

git diff coverage-improvement master

## Self-assessment: Way of working

We have progressed significantly and are now very close to the Working Well state, as the practices and tools 
required for our work have become increasingly natural for the team to use. The team has developed strong 
familiarity with the tools, and most processes flow smoothly with minimal conscious effort. While we still 
occasionally encounter minor friction points in our workflow, these instances are becoming increasingly rare. 
The team has demonstrated good adaptability in fine-tuning our way-of-working, though there remains some room 
for optimization in how we integrate certain practices. The few remaining obstacles to fully reaching the 
Working Well state include occasional misalignments between tools and specific workflow scenarios, and some 
team members still working on mastering advanced features of our tools. These minor challenges are being 
actively addressed through ongoing knowledge sharing and incremental process improvements.


## Overall experience

Our team's experience with the Keras project provided valuable insights into managing and improving complex codebases. The initial onboarding presented some challenges with dependency management, particularly for macOS users, but these were resolved through version adjustments. Our analysis of code complexity revealed several high-complexity functions, which we successfully refactored to improve maintainability and testability. The coverage analysis and improvement efforts yielded mixed results, with coverage improvements ranging from modest (1% for update_confusion_matrix_variables) to significant (24% for deserialize_keras_object). Working with coverage tools was straightforward, though our custom tool implementation highlighted the trade-offs between manual and automatic instrumentation. Throughout the project, our team's way of working has evolved positively, reaching a point where we're effectively using tools and practices with only minor friction points remaining. Despite some challenges, the project has been a successful learning experience in understanding and improving large-scale open-source software.
