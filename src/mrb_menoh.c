#include <menoh/menoh.h>

#include <mruby.h>
#include <mruby/array.h>
#include <mruby/class.h>
#include <mruby/data.h>
#include <mruby/string.h>

static void
check_error(mrb_state *mrb, menoh_error_code code)
{
  if (code == menoh_error_code_success) { return; }
  mrb_raise(mrb, mrb_class_get(mrb, "MenohError"), menoh_get_last_error_message());
}

static void
model_data_free(mrb_state *mrb, void *ptr)
{
  menoh_model_data_handle h = (menoh_model_data_handle)ptr;
  if (h) {
    menoh_delete_model_data(h);
    mrb_free(mrb, ptr);
  }
}

static mrb_data_type model_data_type = { "Menoh::ModelData", model_data_free };

static void
profile_free(mrb_state *mrb, void *ptr)
{
  menoh_variable_profile_table_handle h = (menoh_variable_profile_table_handle)ptr;
  if (h) {
    menoh_delete_variable_profile_table(h);
  }
}

static mrb_data_type profile_type = { "Menoh::VariableProfileTable", profile_free };

static menoh_dtype
to_dtype(mrb_state *mrb, mrb_value v)
{
  if (mrb_symbol_p(v) && mrb_symbol(v) == mrb_intern_lit(mrb, "float")) {
    return menoh_dtype_float;
  }

  mrb_raisef(mrb, E_TYPE_ERROR, "invalid data type name: %S", v);
  return 0;
}

static mrb_value
model_data_from_onnx_file(mrb_state *mrb, mrb_value self)
{
  char const *fn;
  menoh_model_data_handle h;
  mrb_get_args(mrb, "z", &fn);
  check_error(mrb, menoh_make_model_data_from_onnx(fn, &h));
  return mrb_obj_value(mrb_data_object_alloc(mrb, mrb_class_ptr(self), h, &model_data_type));
}

static mrb_value
model_data_from_onnx_memory(mrb_state *mrb, mrb_value self)
{
  mrb_value str;
  menoh_model_data_handle h;
  mrb_get_args(mrb, "S", &str);
  check_error(mrb, menoh_make_model_data_from_onnx_data_on_memory((const uint8_t*)RSTRING_PTR(str), RSTRING_LEN(str), &h));
  return mrb_obj_value(mrb_data_object_alloc(mrb, mrb_class_ptr(self), h, &model_data_type));
}

static mrb_value
model_data_add_parameter(mrb_state *mrb, mrb_value self)
{
  char const *name;
  mrb_value dtype;
  mrb_value dims_v, buffer;
  mrb_get_args(mrb, "zoAS", &name, &dtype, &dims_v, &buffer);
  int32_t dims[RARRAY_LEN(dims_v)];
  for (int i = 0; i < RARRAY_LEN(dims_v); ++i) {
    dims[i] = mrb_fixnum(RARRAY_PTR(dims_v)[i]);
  }
  check_error(mrb, menoh_model_data_add_parameter((menoh_model_data_handle)DATA_PTR(self), name, to_dtype(mrb, dtype), RARRAY_LEN(dims_v), dims, RSTRING_PTR(buffer)));
  return self;
}

static mrb_value
model_data_add_new_node(mrb_state *mrb, mrb_value self)
{
  char const *name;
  mrb_get_args(mrb, "z", &name);
  check_error(mrb, menoh_model_data_add_new_node((menoh_model_data_handle)DATA_PTR(self), name));
  return self;
}

static mrb_value
model_data_add_input_name(mrb_state *mrb, mrb_value self)
{
  char const *name;
  mrb_get_args(mrb, "z", &name);
  check_error(mrb, menoh_model_data_add_input_name_to_current_node((menoh_model_data_handle)DATA_PTR(self), name));
  return self;
}

static mrb_value
model_data_add_output_name(mrb_state *mrb, mrb_value self)
{
  char const *name;
  mrb_get_args(mrb, "z", &name);
  check_error(mrb, menoh_model_data_add_output_name_to_current_node((menoh_model_data_handle)DATA_PTR(self), name));
  return self;
}

static mrb_value
model_data_add_attribute_int(mrb_state *mrb, mrb_value self)
{
  char const *name;
  mrb_int v;
  mrb_get_args(mrb, "zi", &name, &v);
  check_error(mrb, menoh_model_data_add_attribute_int_to_current_node((menoh_model_data_handle)DATA_PTR(self), name, v));
  return self;
}

static mrb_value
model_data_add_attribute_float(mrb_state *mrb, mrb_value self)
{
  char const *name;
  mrb_float v;
  mrb_get_args(mrb, "zf", &name, &v);
  check_error(mrb, menoh_model_data_add_attribute_float_to_current_node((menoh_model_data_handle)DATA_PTR(self), name, v));
  return self;
}

static mrb_value
model_data_optimize(mrb_state *mrb, mrb_value self)
{
  menoh_variable_profile_table_handle prof;
  mrb_get_args(mrb, "d", &prof, &profile_type);
  check_error(mrb, menoh_model_data_optimize((menoh_model_data_handle)DATA_PTR(self), prof));
  return self;
}

static void
model_builder_free(mrb_state *mrb, void *ptr)
{
  menoh_model_builder_handle builder = (menoh_model_builder_handle)ptr;
  if (builder) {
    menoh_delete_model_builder(builder);
  }
}

static mrb_data_type model_builder_type = { "Menoh::ModelBuilder", model_builder_free };

static mrb_value
model_builder_init(mrb_state *mrb, mrb_value self)
{
  menoh_variable_profile_table_handle prof;
  menoh_model_builder_handle h;
  mrb_get_args(mrb, "d", &prof, &profile_type);
  check_error(mrb, menoh_make_model_builder(prof, &h));
  mrb_data_init(self, h, &model_builder_type);
  return self;
}

static mrb_value
model_builder_attach_external_buffer(mrb_state *mrb, mrb_value self)
{
  char const *name;
  mrb_value buf;
  mrb_get_args(mrb, "zS", &name, &buf);
  check_error(mrb, menoh_model_builder_attach_external_buffer((menoh_model_builder_handle)DATA_PTR(self), name, RSTRING_PTR(buf)));
  return self;
}

static void
model_free(mrb_state *mrb, void *ptr)
{
  menoh_model_handle h = (menoh_model_handle)ptr;
  if (h) {
    menoh_delete_model(h);
  }
}

static mrb_data_type model_type = { "Menoh::Model", model_free };

static mrb_value
model_init(mrb_state *mrb, mrb_value self)
{
  menoh_model_builder_handle builder;
  menoh_model_data_handle data;
  char const *backend_name, *backend_cfg;
  menoh_model_handle model;
  mrb_get_args(mrb, "ddzz", &builder, &model_builder_type, &data, &model_data_type, &backend_name, &backend_cfg);
  check_error(mrb, menoh_build_model(builder, data, backend_name, backend_cfg, &model));
  mrb_data_init(self, model, &model_type);
  return self;
}

void
mrb_mruby_menoh_gem_init(mrb_state *mrb)
{
  struct RClass* mod = mrb_define_module(mrb, "Menoh");
  struct RClass* model_data = mrb_define_class_under(mrb, mod, "ModelData", mrb->object_class);
  struct RClass* model_builder = mrb_define_class_under(mrb, mod, "ModelBuilder", mrb->object_class);
  struct RClass* model = mrb_define_class_under(mrb, mod, "Model", mrb->object_class);
  struct RClass* profile = mrb_define_class_under(mrb, mod, "VariableProfileTable", mrb->object_class);
  struct RClass* profile_builder = mrb_define_class_under(mrb, mod, "VariableProfileTableBuilder", mrb->object_class);

  MRB_SET_INSTANCE_TT(model_data, MRB_TT_DATA);
  MRB_SET_INSTANCE_TT(model_builder, MRB_TT_DATA);
  MRB_SET_INSTANCE_TT(model, MRB_TT_DATA);
  MRB_SET_INSTANCE_TT(profile, MRB_TT_DATA);
  MRB_SET_INSTANCE_TT(profile_builder, MRB_TT_DATA);

  mrb_undef_method(mrb, model_data, "initialize");
  mrb_define_class_method(mrb, model_data, "from_onnx_file", model_data_from_onnx_file, MRB_ARGS_REQ(1));
  mrb_define_class_method(mrb, model_data, "from_onnx_memory", model_data_from_onnx_memory, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, model_data, "add_parameter", model_data_add_parameter, MRB_ARGS_REQ(3));
  mrb_define_method(mrb, model_data, "add_new_node", model_data_add_new_node, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, model_data, "add_input_name_to_current_node", model_data_add_input_name, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, model_data, "add_output_name_to_current_node", model_data_add_output_name, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, model_data, "add_attribute_int_to_current_node", model_data_add_attribute_int, MRB_ARGS_REQ(2));
  mrb_define_method(mrb, model_data, "add_attribute_float_to_current_node", model_data_add_attribute_float, MRB_ARGS_REQ(2));
  mrb_define_method(mrb, model_data, "optimize", model_data_optimize, MRB_ARGS_REQ(1));

  mrb_define_method(mrb, model_builder, "initialize", model_builder_init, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, model_builder, "attach_external_buffer", model_builder_attach_external_buffer, MRB_ARGS_REQ(2));

  mrb_define_method(mrb, model, "initialize", model_init, MRB_ARGS_REQ(3));
  mrb_define_method(mrb, model, "variable_buffer_handle", model_variable_buffer_handle, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, model, "variable_dtype", model_variable_dtype, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, model, "variable_dims", model_variable_dims, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, model, "run", MRB_ARGS_NONE());

  mrb_define_method(mrb, profile, "initialize", prof_init, MRB_ARGS_REQ(2));
  mrb_define_method(mrb, profile, "dtype", prof_dtype, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, profile, "dims", prof_dims, MRB_ARGS_REQ(1));

  mrb_define_method(mrb, profile_builder, "initialize", profile_builder_init, MRB_ARGS_NONE());
  mrb_define_method(mrb, profile_builder, "add_input_profile", profile_builder_add_input_profile, MRB_ARGS_REQ(3));
  mrb_define_method(mrb, profile_builder, "add_input_profile_dims_2", profile_builder_add_input_profile_dims_2, MRB_ARGS_REQ(4));
  mrb_define_method(mrb, profile_builder, "add_input_profile_dims_4", profile_builder_add_input_profile_dims_4, MRB_ARGS_REQ(6));
  mrb_define_method(mrb, profile_builder, "add_output_name", profile_builder_add_output_name, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, profile_builder, "add_output_profile", profile_builder_add_output_profile, MRB_ARGS_REQ(2));
}

void
mrb_mruby_menoh_gem_final(mrb_state *mrb)
{
}
