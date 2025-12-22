# frozen_string_literal: true

require 'matrix'
class NeuralNetwork
  WEIGHT_INPUT_HIDDEN = Matrix[
    [0.143827, -0.13728512, 0.24625022],
    [0.41129188, 0.24551424, -0.43500754],
    [0.3188901, 0.06173198, 0.18406137]
  ]
  WEIGHT_HIDDEN_OUTPUT = Matrix[
    [0.028071, 0.2437462, -0.158728],
    [0.0673178, -0.01818, 0.46627],
    [0.390290, -0.2669001, 0.180840]]

  def initialize(
    input_nodes_count:,
    hidden_nodes_count:,
    output_nodes_count:,
    learning_rate:
  )
    @input_nodes_count = input_nodes_count
    @hidden_nodes_count = hidden_nodes_count
    @output_nodes_count = output_nodes_count
    @learning_rate = learning_rate
  end

  def train; end

  def query(input_list:)
    # convert input_list to a matrix
    inputs = Matrix.column_vector(input_list)
    hidden_inputs = WEIGHT_INPUT_HIDDEN * inputs
    hidden_outputs = activation_function(hidden_inputs)
    final_inputs = WEIGHT_HIDDEN_OUTPUT * hidden_outputs

    activation_function(final_inputs)
  end

  private

  def activation_function(input_matrix)
    Matrix.column_vector(
      input_matrix.column(0).map do |input|
        1.0 / (1.0 + Math.exp(-input))
      end
    )
  end
end
