# frozen_string_literal: true

require_relative '../neural_network'
require 'benchmark'
require 'numo/narray'

describe NeuralNetwork do
  describe '#activation_function' do
    it 'applies the activation function to a matrix' do
      neural_network = NeuralNetwork.new(
        input_nodes_count: 3,
        hidden_nodes_count: 3,
        output_nodes_count: 3,
        learning_rate: 0.2
      )

      matrix = Numo::DFloat[[1.05], [0.6]]

      expect(neural_network.activation_function(matrix)[0, 0]).to be_within(0.0001).of(0.7408)
      expect(neural_network.activation_function(matrix)[1, 0]).to be_within(0.0001).of(0.6457)
    end
  end

  describe '#query' do
    it 'calculates the output of an input' do
      neural_network = NeuralNetwork.new(
        input_nodes_count: 3,
        hidden_nodes_count: 3,
        output_nodes_count: 3,
        learning_rate: 0.2
      )

      result = neural_network.query(input_list: Numo::DFloat[[1], [2], [3]])
      expect(result[0, 0]).to be_within(0.5).of(0.5)
      expect(result[1, 0]).to be_within(0.5).of(0.5)
      expect(result[2, 0]).to be_within(0.5).of(0.5)
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

      expect(network.weight_input_hidden.shape).to eq([6, 12])
      expect(network.weight_hidden_output.shape).to eq([3, 6])
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

      inputs = Numo::NArray[[0.55], [0.33], [0.22]]
      targets = Numo::NArray[[0.99]]

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
        Numo::NArray[
          [2, 3],
          [12, 52]
        ]
      )

      expect(neural_network.weight_hidden_output).to eq(
        Numo::NArray[
          [5], [9]
        ]
      )
    end
  end

  describe '#calculate_weights_after_applying_error' do
    it 'calculates the new weights for a layer, based on the error' do
      input_weights = Numo::NArray[[2.0, 3.0]]
      previous_layer_outputs = Numo::NArray[[0.4], [0.5]]
      outputs = Numo::NArray[[0.909]]
      errors = Numo::NArray[[0.8]]
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
      )

      expect(
        new_weights[0]
      ).to be_within(0.000001).of(2.002647008)

      expect(
        new_weights[1]
      ).to be_within(0.000001).of(3.00330876)
    end

    it 'runs as fast as it can' do
      # previous time for 60_000 reps: 16.1s
      # previous time for 60_000 reps: 0.37s
      input_weights = Numo::NArray[Array.new(768, rand)]
      previous_layer_outputs = Numo::NArray[Array.new(768, rand)].transpose
      outputs = Numo::NArray[[0.909]]
      errors = Numo::NArray[[0.8]]
      learning_rate = 0.1

      network = NeuralNetwork.new(
        input_nodes_count: 3,
        hidden_nodes_count: 3,
        output_nodes_count: 1,
        learning_rate: learning_rate
      )
      elapsed_time = Benchmark.measure do
        60_000.times do
          network.calculate_weights_after_applying_error(
            previous_layer_outputs: previous_layer_outputs,
            input_weights: input_weights,
            outputs: outputs,
            errors: errors
          )
        end
      end

      expect(elapsed_time.real).to be_within(0.3).of(0.3)
    end
  end

  # describe 'Matrix methods' do
  #   require 'benchmark'

  #   it 'takes a certain amount of time to do calculations by mapping on a matrix' do
  #     matrix = Matrix.build(1000, 1) { 5 }
  #     elapsed_time = Benchmark.measure do
  #       60_000.times do
  #         matrix.map { |item| 1 - item }
  #       end
  #     end

  #     puts "Mapping through elements: #{elapsed_time}"
  #   end

  #   it 'takes another amount of time to do calculations by instantiating a new Vector' do
  #     matrix = Matrix.build(1000, 1) { 5 }
  #     elapsed_time = Benchmark.measure do
  #       vector = Vector.elements(Array.new(1000, 1))
  #       60_000.times do
  #         vector - matrix.column_vectors.first
  #       end
  #     end

  #     puts "Using the extra matrix method: #{elapsed_time}"
  #   end
  # end
end
