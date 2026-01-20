require_relative './neural_network'
require_relative './input_processor'
require 'json'

network = NeuralNetwork.new(
  # we are working with 28 x 28 pixel images
  input_nodes_count: 784,
  hidden_nodes_count: 200,
  output_nodes_count: 10,
  learning_rate: 0.1
)

training_data = InputProcessor.new("#{__dir__}/../MNIST_CSV/mnist_train.csv").processed_data
# training_data = InputProcessor.new("#{__dir__}/spec/fixtures/mnist_100_items.csv").processed_data

pretraining_file_name = "#{__dir__}/trained_weights.json"
if File.exist?(pretraining_file_name)
  network.load_pretrained_weights(pretraining_file_name)
else
  2.times do |count|
    puts "Training number: #{count}"

    training_data.each_with_index do |row_data, record_index|
      inputs = row_data[:data]
      targets = Matrix.zero(network.output_nodes_count, 1).map { |element| element + 0.01 }
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
end

test_data = []
InputProcessor.new("#{__dir__}/spec/fixtures/mnist_10_items.csv").read_csv_data { |row| test_data << row}

reports = []
test_data.each do |image_data|
  report = { actual: image_data[:label] }
  output = network.query(input_list: image_data[:data]).column_vectors.first.to_a
  predicted = output.index(output.max).to_s
  report[:predicted] = predicted
  report[:prediction_was_correct] = predicted == image_data[:label]
  reports << report
end

puts reports
