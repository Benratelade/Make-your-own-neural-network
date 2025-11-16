# frozen_string_literal: true

class NeuralNetwork
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

  def init; end
  def train; end
  def query; end
end
