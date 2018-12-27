MRuby::Gem::Specification.new 'mruby-menoh' do |spec|
  spec.license = 'MIT'
  spec.author = 'Takeshi Watanabe'

  spec.linker.libraries << 'menoh'
end
