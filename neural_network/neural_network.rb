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
    @activation_function_cache = {}
    @one_vector_cache = {}
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

    # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
    hidden_errors = @weight_hidden_output.transpose * output_errors

    # update the weights for the links between the hidden and output layers
    @weight_hidden_output = calculate_weights_after_applying_error(
      previous_layer_outputs: hidden_outputs,
      input_weights: @weight_hidden_output,
      outputs: final_outputs,
      errors: output_errors
    )

    # update the weights for the links between the input and hidden layers
    @weight_input_hidden = calculate_weights_after_applying_error(
      previous_layer_outputs: inputs,
      input_weights: @weight_input_hidden,
      outputs: hidden_outputs,
      errors: hidden_errors
    )
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

  def activation_function(input_matrix)
    Matrix.column_vector(
      input_matrix.column(0).map do |input|
        result = @activation_function_cache[input] || 1.0 / (1.0 + Math.exp(-input))
        @activation_function_cache[input] ||= result
      end
    )
  end

  def calculate_weights_after_applying_error(previous_layer_outputs:, input_weights:, outputs:, errors:)
    one_vector = @one_vector_cache[errors.row_size] || Vector.elements(Array.new(errors.row_size, 1))
    @one_vector_cache[errors.row_size] ||= one_vector
    error_applied_to_outputs = errors.entrywise_product(
      outputs.entrywise_product(Matrix.column_vector(one_vector - outputs.column_vectors.first))
    )
    input_weights + (@learning_rate * (error_applied_to_outputs * previous_layer_outputs.transpose))
  end

  private

  def generate_starting_weights_for_network
    @weight_input_hidden = Matrix.build(@hidden_nodes_count, @input_nodes_count) { Random.rand - 0.5 }
    @weight_hidden_output = Matrix.build(@output_nodes_count, @hidden_nodes_count) { Random.rand - 0.5 }
  end
end
