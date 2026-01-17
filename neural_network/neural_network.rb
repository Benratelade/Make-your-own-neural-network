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

  attr_reader :output_nodes_count,
              :weight_input_hidden,
              :weight_hidden_output

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
    generate_starting_weights_for_network
  end

  def train(inputs:, targets:)
    # convert inputs list to 2d array
    inputs = Matrix[inputs]
    targets = Matrix[targets]

    # calculate signals into hidden layer
    hidden_inputs = inputs * WEIGHT_INPUT_HIDDEN

    # calculate the signals emerging from hidden layer
    hidden_outputs = activation_function(hidden_inputs)
    # calculate signals into final output layer
    final_inputs = hidden_outputs * WEIGHT_HIDDEN_OUTPUT

    # calculate the signals emerging from final output layer
    final_outputs = activation_function(final_inputs)

    # output layer error is the (target - actual)
    output_errors = targets - final_outputs

    # hidden layer error is the output_errors, split by weights,  recombined at hidden nodes
    hidden_errors = @weight_hidden_output * output_errors

    # update the weights for the links between the hidden and  output layers
    @weight_hidden_output += @learning_rate *
                             ((output_errors * final_outputs * (1.0 - final_outputs)) * hidden_outputs)

    # update the weights for the links between the input and  hidden layers
    @weight_input_hidden += @learning_rate *
                            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs) * inputs)
  end

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

  def generate_starting_weights_for_network
    @weight_input_hidden = Matrix.build(@hidden_nodes_count, @input_nodes_count) { rand - 0.5 }
    @weight_hidden_output = Matrix.build(@output_nodes_count, @hidden_nodes_count) { rand - 0.5 }
  end
end
