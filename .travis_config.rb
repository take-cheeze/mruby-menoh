MRuby::Build.new do |conf|
  toolchain :gcc
  enable_debug
  enable_test

  gem "#{MRUBY_ROOT}/.."
end
