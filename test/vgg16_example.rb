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

  image_data = load_image_data input_image_path, width, height

  model_data = Menoh::ModelData.from_onnx_file onnx_model_path

  vpt_builder = Menoh::VariableProfileTableBuilder.new
  vpt_builder.add_input_profile conv1_1_in_name, :float, [batch_size, channel_num, height, width]

  vpt = Menoh::VariableProfileTable.new model_data
  fc6_dims = vpt.variable_dims(fc6_out_name)
  fc6_out_data = fc6_dims.pack('f' * fc6_dims.size)

  model_data.optimize vpt

  model_builder = Menoh::ModelBuilder.new vpt
  model_builder.attach_external_buffer conv1_1_in_name, image_data
  model_builder.attach_external_buffer fc6_out_name, fc6_out_data

  model = Menoh::Model.new(model_data, "composite_backend", '{"backends":[{"type":"mkldnn"}, {"type":"generic"}], "log_output":"stdout"}')

  softmax_output_dims = model.variable_buffer_dims softmax_out_name
  softmax_output_buf = model.variable_buffer_handle softmax_out_name

  model.run

  __t_printstr__ fc6_out_data.unpack('f*').inspect + "\n"

  categories = File.read(synset_words_path).split("\n")
  top_k = 5
  top_k_indices = extract_top_k_index_list(softmax_output_buf, softmax_output_buf + softmax_output_dims[1], top_k)

  top_k_indices.each do |v|
    __t_printstr__ categories[v],inspect + "\n"
  end
end
