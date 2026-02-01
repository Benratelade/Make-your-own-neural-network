# frozen_string_literal: true

require_relative '../input_processor'
require 'matrix'

describe InputProcessor do
  describe '#stream_csv_data'
  it 'reads and yields the CSV data in a format the neural network can use' do
    input_processor = InputProcessor.new("#{__dir__}/fixtures/mnist_10_items.csv")

    pre_processed_data = []
    input_processor.stream_csv_data do |data|
      pre_processed_data << data
    end

    expect(pre_processed_data.map { |row| row[:label] }.flatten).to eq(%w[7 2 1 0 4 1 4 9 5 9])
    expect(pre_processed_data.first[:data].size).to eq(28**2)
  end
end
