# frozen_string_literal: true

require_relative './neural_network'
require_relative './input_processor'

puts 'START generating the neural network'
network = NeuralNetwork.new(
  # we are working with 28 x 28 pixel images
  input_nodes_count: 784,
  hidden_nodes_count: 200,
  output_nodes_count: 10,
  learning_rate: 0.1
)
puts 'END generating the neural network'

pretraining_file_name = "#{__dir__}/trained_weights.json"
network.load_pretrained_weights(pretraining_file_name) if File.exist?(pretraining_file_name)

test_data = InputProcessor.new("#{__dir__}/../MNIST_CSV/mnist_test.csv").processed_data

reports = []
test_data.each do |image_data|
  report = { actual: image_data[:label] }
  output = network.query(input_list: image_data[:data]).column_vectors.first.to_a
  predicted = output.index(output.max).to_s
  report[:predicted] = predicted
  report[:prediction_was_correct] = predicted == image_data[:label]
  reports << report
end

report_summary = {}
accurate_predictions = reports.select { |report| report[:prediction_was_correct] }.count
incorrect_predictions = reports.reject { |report| report[:prediction_was_correct] }.count
report_summary[:accurate_predictions] = accurate_predictions
report_summary[:incorrect_predictions] = incorrect_predictions
report_summary[:total_predictions] = reports.count
report_summary[:percentage_correct] = (accurate_predictions.to_f / reports.count) * 100
puts report_summary
