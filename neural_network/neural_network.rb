# frozen_string_literal: true

require 'matrix'
require 'json'
class NeuralNetwork
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
    # calculate signals into hidden layer
    hidden_inputs = @weight_input_hidden * inputs

    # calculate the signals emerging from hidden layer
    hidden_outputs = activation_function(hidden_inputs)

    # calculate signals into final output layer
    final_inputs = @weight_hidden_output * hidden_outputs

    # calculate the signals emerging from final output layer
    final_outputs = activation_function(final_inputs)

    # output layer error is the (target - actual)
    output_errors = targets - final_outputs

    # hidden layer error is the output_errors, split by weights,  recombined at hidden nodes
    hidden_errors = @weight_hidden_output.transpose * output_errors

    # update the weights for the links between the hidden and  output layers
    @weight_hidden_output += @learning_rate *
                             (output_errors.combine(final_outputs.map do |el|
                               el * (1.0 - el)
                             end) { |error, output| error * output } * hidden_outputs.transpose)

    # update the weights for the links between the input and  hidden layers
    @weight_input_hidden += @learning_rate *
                            (hidden_errors.combine(hidden_outputs.map { |el| el * (1.0 - el) }) do |error, output|
                              error * output
                            end * inputs.transpose)
  end

  def query(input_list:)
    # Assume the input list is already a matrix
    hidden_inputs = @weight_input_hidden * input_list
    hidden_outputs = activation_function(hidden_inputs)
    final_inputs = @weight_hidden_output * hidden_outputs

    activation_function(final_inputs)
  end

  def load_pretrained_weights(file)
    data = JSON.parse(File.read(file))
    @weight_input_hidden = Matrix[*data['weight_input_hidden']]
    @weight_hidden_output = Matrix[*data['weight_hidden_output']]
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
