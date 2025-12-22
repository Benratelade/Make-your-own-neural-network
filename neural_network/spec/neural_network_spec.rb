# frozen_string_literal: true

require_relative '../neural_network'
require 'matrix'

describe NeuralNetwork do
  it 'calculates the output of an input' do
    neural_network = NeuralNetwork.new(
      input_nodes_count: 3,
      hidden_nodes_count: 3,
      output_nodes_count: 3,
      learning_rate: 0.2
    )

    expect(neural_network.query(input_list: [1, 2, 3])).to eq(
      Matrix[[0.4999917331524737], [0.5930703309561164], [0.5690019973981506]]
    )
  end
end
