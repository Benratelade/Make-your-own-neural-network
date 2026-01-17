require 'csv'
require 'matrix'

class InputProcessor
  attr_reader :processed_data

  def initialize(input_file)
    @input_file = input_file
    @processed_data = []
    read_csv_data
  end

  def read_csv_data
    CSV.read(@input_file).each do |csv|
      @processed_data << {
        label: csv.first,
        data: convert_and_rescale_data(csv[1..])
      }
    end
  end

  def convert_and_rescale_data(data)
    matrix = Matrix[data.map(&:to_i)]
    (matrix / 255) * 0.99
  end
end
