# frozen_string_literal: true

require_relative '../input_processor'
require 'matrix'

describe InputProcessor do
  describe '#read_csv_data'
  it 'reads and converts the CSV data to something the neural network can use' do
    input_processor = InputProcessor.new("#{__dir__}/fixtures/mnist_10_items.csv")

    expect(input_processor.processed_data.map { |row| row[:label] }.flatten).to eq(%w[7 2 1 0 4 1 4 9 5 9])
    expect(input_processor.processed_data.first[:data].column_size).to eq(28**2)
  end
end
