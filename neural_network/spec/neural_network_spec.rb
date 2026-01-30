# frozen_string_literal: true

require_relative '../neural_network'
require 'matrix'

describe NeuralNetwork do
  describe '#activation_function' do
    it 'applies the activation function to a matrix' do
      neural_network = NeuralNetwork.new(
        input_nodes_count: 3,
        hidden_nodes_count: 3,
        output_nodes_count: 3,
        learning_rate: 0.2
      )

      matrix = Matrix[[1.05], [0.6]]

      expect(neural_network.activation_function(matrix)[0, 0]).to be_within(0.0001).of(0.7408)
      expect(neural_network.activation_function(matrix)[1, 0]).to be_within(0.0001).of(0.6457)
    end
  end

  describe '#query' do
    it 'calculates the output of an input' do
      allow(Random).to receive(:rand).and_return(3)
      neural_network = NeuralNetwork.new(
        input_nodes_count: 3,
        hidden_nodes_count: 3,
        output_nodes_count: 3,
        learning_rate: 0.2
      )

      expect(neural_network.query(input_list: Matrix[[1], [2], [3]])).to eq(
        Matrix[[0.9994472200955544], [0.9994472200955544], [0.9994472200955544]]
      )
    end
  end

  describe '#generate_starting_weights_for_network' do
    it 'generates random weights for the layers, adapted to the number of nodes' do
      network = NeuralNetwork.new(
        input_nodes_count: 12,
        hidden_nodes_count: 6,
        output_nodes_count: 3,
        learning_rate: 0.2
      )

      expect(network.weight_input_hidden.column_size).to eq(12)
      expect(network.weight_input_hidden.row_size).to eq(6)
      expect(network.weight_hidden_output.column_size).to eq(6)
      expect(network.weight_hidden_output.row_size).to eq(3)
    end
  end

  describe '#train' do
    it 'changes the weights to try and reach the targets' do
      network = NeuralNetwork.new(
        input_nodes_count: 3,
        hidden_nodes_count: 2,
        output_nodes_count: 1,
        learning_rate: 0.2
      )

      inputs = Matrix[[0.55], [0.33], [0.22]]
      targets = Matrix[[0.99]]

      expect do
        network.train(inputs: inputs, targets: targets)
      end.to change(network, :weight_input_hidden)

      expect do
        network.train(inputs: inputs, targets: targets)
      end.to change(network, :weight_hidden_output)
    end
  end

  describe '#load_pretrained_weights' do
    before do
    end
    it 'loads pretrained weights from a file' do
      neural_network = NeuralNetwork.new(
        input_nodes_count: 2,
        hidden_nodes_count: 2,
        output_nodes_count: 1,
        learning_rate: 0.2
      )
      neural_network.load_pretrained_weights("#{__dir__}/fixtures/pretrained_weights.json")

      expect(neural_network.weight_input_hidden).to eq(
        Matrix[
          [2, 3],
          [12, 52]
        ]
      )

      expect(neural_network.weight_hidden_output).to eq(
        Matrix[
          [5], [9]
        ]
      )
    end
  end

  describe '#calculate_weights_after_applying_error' do
    it 'calculates the new weights for a layer, based on the error' do
      input_weights = Matrix[[2.0, 3.0]]
      previous_layer_outputs = Matrix[[0.4], [0.5]]
      outputs = Matrix[[0.909]]
      errors = Matrix[[0.8]]
      learning_rate = 0.1

      network = NeuralNetwork.new(
        input_nodes_count: 3,
        hidden_nodes_count: 3,
        output_nodes_count: 1,
        learning_rate: learning_rate
      )

      new_weights = network.calculate_weights_after_applying_error(
        previous_layer_outputs: previous_layer_outputs,
        input_weights: input_weights,
        outputs: outputs,
        errors: errors
      ).row_vectors.first

      expect(
        new_weights[0]
      ).to be_within(0.000001).of(2.002647008)

      expect(
        new_weights[1]
      ).to be_within(0.000001).of(3.00330876)
    end
  end
end
