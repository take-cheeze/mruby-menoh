MRuby::Gem::Specification.new 'mruby-menoh' do |spec|
  spec.license = 'MIT'
  spec.author = 'Takeshi Watanabe'

  search_package 'freetype2'
  search_package 'harfbuzz'
  search_package 'opencv'

  linker.libraries << 'menoh'

  add_test_dependency 'mruby-pack'
  add_test_dependency 'mruby-io'

  `wget https://preferredjp.box.com/shared/static/o2xip23e3f0knwc5ve78oderuglkf2wt.onnx -O #{MRUBY_ROOT}/vgg16.onnx` unless File.exists? "#{MRUBY_ROOT}/vgg16.onnx"
  `wget https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt -O #{MRUBY_ROOT}/synset_words.txt` unless
    File.exists? "#{MRUBY_ROOT}/synset_words.txt"
  `wget https://upload.wikimedia.org/wikipedia/commons/5/54/Light_sussex_hen.jpg -O #{MRUBY_ROOT}/Light_sussex_hen.jpg` unless
    File.exists? "#{MRUBY_ROOT}/Light_sussex_hen.jpg"
end
