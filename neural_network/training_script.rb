require_relative './neural_network'
require_relative './input_processor'
require 'numo/narray'
require 'json'

puts 'START generating the neural network'
network = NeuralNetwork.new(
  # we are working with 28 x 28 pixel images
  input_nodes_count: 784,
  hidden_nodes_count: 200,
  output_nodes_count: 10,
  learning_rate: 0.1
)
puts 'END generating the neural network'

puts 'START processing data'
training_file = "#{__dir__}/../MNIST_CSV/mnist_train.csv"
# training_file = "#{__dir__}/spec/fixtures/mnist_100_items.csv"
data_processor = InputProcessor.new(training_file)
puts 'END processing data'

3.times do |count|
  puts "Starting Epoch: #{count + 1}"

  record_index = 0
  data_processor.read_csv_data do |row_data|
    inputs = row_data[:data]
    targets = Numo::DFloat.new(network.output_nodes_count, 1).fill(0.01)
    targets[row_data[:label].to_i, 0] = 0.99
    network.train(inputs: inputs, targets: targets)
    record_index += 1

    puts "trained #{record_index}" if (record_index % 1000).zero?
  end
end

File.open('neural_network/trained_weights.json', 'w') do |file|
  file.puts(
    {
      weight_input_hidden: network.weight_input_hidden.to_a,
      weight_hidden_output: network.weight_hidden_output.to_a
    }.to_json
  )
end

test_data_processor = InputProcessor.new("#{__dir__}/spec/fixtures/mnist_10_items.csv")

reports = []
test_data_processor.read_csv_data do |image_data|
  report = { actual: image_data[:label] }
  output = network.query(input_list: image_data[:data])
  predicted = output[0.., 0].to_a.index(output.max).to_s
  report[:predicted] = predicted
  report[:prediction_was_correct] = predicted == image_data[:label]
  reports << report
end

puts reports
