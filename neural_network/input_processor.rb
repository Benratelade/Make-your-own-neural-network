require 'csv'
require 'numo/narray'

class InputProcessor
  attr_reader :processed_data

  def initialize(input_file)
    @input_file = input_file
    @processed_data = []
    # stream_csv_data
  end

  def stream_csv_data
    CSV.foreach(@input_file).each do |row|
      yield(
        {
          label: row.first, data: convert_and_rescale_data(row[1..])
        }
      )
    end
  end

  def convert_and_rescale_data(data)
    matrix = Numo::DFloat[data.map(&:to_f)]
    (((matrix / 255) * 0.99) + 0.01).transpose
  end
end
