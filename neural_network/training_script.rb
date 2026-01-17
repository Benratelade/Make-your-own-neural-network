require_relative './neural_network'
require_relative './input_processor'

network = NeuralNetwork.new(
  # we are working with 28 x 28 pixel images
  input_nodes_count: 784,
  hidden_nodes_count: 200,
  output_nodes_count: 10,
  learning_rate: 0.1
)

training_data = InputProcessor.new("#{__dir__}/spec/fixtures/mnist_100_items.csv").processed_data

5.times do
  training_data.each do |record|
    inputs = record[:data]
    targets = Matrix.zero(network.output_nodes_count, 1).map { |element| element + 0.01 }
    targets[record[:label].to_i, 0] = 0.99
    network.train(inputs: inputs, targets: targets)
  end
end
