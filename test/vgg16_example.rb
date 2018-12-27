assert 'VGG16' do
  conv1_1_in_name = "Input_0"
  fc6_out_name = "Gemm_0"
  softmax_out_name = "Softmax_0"

  batch_size = 1
  channel_num = 3
  height = 224
  width = 224

  input_image_path = 'Light_sussex_hen.jpg'
  onnx_model_path = 'vgg16.onnx'
  synset_words_path = 'synset_words.txt'

  model_data = Menoh::ModelData.from_onnx_file onnx_model_path

  vpt_builder = Menoh::VariableProfileTableBuilder.new
  vpt_builder.add_input_profile conv1_1_in_name, :float, [batch_size, channel_num, height, width]
  vpt_builder.add_output_name fc6_out_name
  vpt_builder.add_output_name softmax_out_name

  vpt = Menoh::VariableProfileTable.new vpt_builder, model_data

  model_data.optimize vpt

  image_data = load_image_data input_image_path, width, height
  fc6_out_data = ([0.0] * vpt.dims(fc6_out_name)[1]).pack('f*')
  softmax_output = ([0.0] * vpt.dims(softmax_out_name)[1]).pack('f*')

  model_builder = Menoh::ModelBuilder.new vpt
  model_builder.attach_external_buffer conv1_1_in_name, image_data
  model_builder.attach_external_buffer fc6_out_name, fc6_out_data
  model_builder.attach_external_buffer softmax_out_name, softmax_output

  model = Menoh::Model.new model_builder, model_data, 'mkldnn', '{}'

  model.run

  categories = File.read(synset_words_path).split("\n")

  scores = []
  softmax_output.unpack('f*').each_with_index do |v, idx|
    scores << [v, categories[idx]]
  end
  scores.sort!{ |a, b| a[0] <=> b[0] }

  assert_equal 'hen', scores.last[1].split[1]
end
