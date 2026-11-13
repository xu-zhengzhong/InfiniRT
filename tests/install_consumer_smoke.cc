#include <infini/rt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

int main() {
  std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
  const infini::rt::Device device{infini::rt::Device::Type::kCpu};
  const std::vector<std::size_t> shape{2, 2};
  const std::vector<std::ptrdiff_t> default_strides{2, 1};
  const std::vector<std::ptrdiff_t> explicit_strides{1, 2};
  const infini::rt::TensorView::Strides filled_strides(shape.size(), 0);
  const infini::rt::TensorView default_view{
      data.data(), shape, infini::rt::DataType::kFloat32, device};
  const infini::rt::TensorView explicit_view{data.data(), shape,
                                             infini::rt::DataType::kFloat32,
                                             device, explicit_strides};

  if (device.ToString() != "cpu:0") {
    return 1;
  }

  if (filled_strides.size() != shape.size() || filled_strides[0] != 0 ||
      filled_strides[1] != 0) {
    return 1;
  }

  if (default_view.numel() != 4 || !default_view.IsContiguous() ||
      default_view.shape() != shape ||
      default_view.strides() != default_strides || default_view.size(-1) != 2 ||
      default_view.stride(-1) != 1) {
    return 1;
  }

  if (explicit_view.numel() != 4 || explicit_view.IsContiguous() ||
      explicit_view.shape() != shape ||
      explicit_view.strides() != explicit_strides ||
      explicit_view.size(0) != 2 || explicit_view.stride(0) != 1) {
    return 1;
  }

  // Exercise sparse metadata through the installed umbrella and shared library.
  // Run before runtime-device checks: descriptor construction needs no backend.
  using infini::rt::DataType;
  using infini::rt::SparseTensorView;
  using infini::rt::TensorLayout;
  using infini::rt::TensorView;
  using infini::rt::ValidationCode;
  using infini::rt::ValidationComponent;
  using infini::rt::ValidationResult;

  std::array<std::int64_t, 4> coo_indices{0, 1, 1, 2};
  std::array<float, 2> sparse_values{2.0f, 3.0f};
  const TensorView indices{
      coo_indices.data(), {2, 2}, DataType::kInt64, device};
  const TensorView values{
      sparse_values.data(), {2}, DataType::kFloat32, device};
  const SparseTensorView::Shape sparse_shape{2, 3};
  std::optional<SparseTensorView> coo;
  const ValidationResult coo_status =
      SparseTensorView::CreateCoo(sparse_shape, indices, values, coo);
  if (!coo_status.ok() || coo_status.code != ValidationCode::kSuccess ||
      coo_status.component != ValidationComponent::kNone ||
      coo_status.position.has_value() || !coo ||
      coo->layout() != TensorLayout::kCoo || coo->shape() != sparse_shape ||
      coo->nnz() != 2 || coo->numel() != 6 || coo->device() != device ||
      coo->dtype() != DataType::kFloat32 ||
      coo->index_dtype() != DataType::kInt64) {
    return 1;
  }
  bool coalesced = true;
  const TensorView* component = nullptr;
  if (!coo->is_coalesced(coalesced) || coalesced ||
      !coo->raw_indices(component) || component == nullptr ||
      component->data() != coo_indices.data() || !coo->raw_values(component) ||
      component == nullptr || component->data() != sparse_values.data()) {
    return 1;
  }
  const ValidationResult access_status = coo->indices(component);
  if (access_status || access_status.code != ValidationCode::kInvalidState ||
      access_status.component != ValidationComponent::kIndices ||
      component != nullptr) {
    return 1;
  }

  std::array<std::int32_t, 3> crow_indices{0, 1, 2};
  std::array<std::int32_t, 2> col_indices{1, 2};
  const TensorView crow{crow_indices.data(), {3}, DataType::kInt32, device};
  const TensorView col{col_indices.data(), {2}, DataType::kInt32, device};
  std::optional<SparseTensorView> csr;
  const ValidationResult csr_status =
      SparseTensorView::CreateCsr({2, 3}, crow, col, values, csr);
  if (!csr_status || !csr || csr->layout() != TensorLayout::kCsr ||
      csr->shape() != sparse_shape || csr->nnz() != 2 ||
      csr->index_dtype() != DataType::kInt32 || !csr->crow_indices(component) ||
      component == nullptr || component->data() != crow_indices.data() ||
      !csr->col_indices(component) || component == nullptr ||
      component->data() != col_indices.data() || !csr->values(component) ||
      component == nullptr || component->data() != sparse_values.data()) {
    return 1;
  }

#if defined(INFINI_RT_CONSUMER_BACKEND_CPU) ||       \
    defined(INFINI_RT_CONSUMER_BACKEND_NVIDIA) ||    \
    defined(INFINI_RT_CONSUMER_BACKEND_ILUVATAR) ||  \
    defined(INFINI_RT_CONSUMER_BACKEND_HYGON) ||     \
    defined(INFINI_RT_CONSUMER_BACKEND_THEAD) ||     \
    defined(INFINI_RT_CONSUMER_BACKEND_METAX) ||     \
    defined(INFINI_RT_CONSUMER_BACKEND_MARS) ||      \
    defined(INFINI_RT_CONSUMER_BACKEND_MOORE) ||     \
    defined(INFINI_RT_CONSUMER_BACKEND_CAMBRICON) || \
    defined(INFINI_RT_CONSUMER_BACKEND_ASCEND)
  namespace runtime = infini::rt::runtime;
#if defined(INFINI_RT_CONSUMER_BACKEND_CPU)
  constexpr auto kExpectedDeviceType = infini::rt::Device::Type::kCpu;
  constexpr bool kExpectAsyncMemcpySuccess = false;
#elif defined(INFINI_RT_CONSUMER_BACKEND_NVIDIA)
  constexpr auto kExpectedDeviceType = infini::rt::Device::Type::kNvidia;
  constexpr bool kExpectAsyncMemcpySuccess = true;
#elif defined(INFINI_RT_CONSUMER_BACKEND_ILUVATAR)
  constexpr auto kExpectedDeviceType = infini::rt::Device::Type::kIluvatar;
  constexpr bool kExpectAsyncMemcpySuccess = true;
#elif defined(INFINI_RT_CONSUMER_BACKEND_HYGON)
  constexpr auto kExpectedDeviceType = infini::rt::Device::Type::kHygon;
  constexpr bool kExpectAsyncMemcpySuccess = true;
#elif defined(INFINI_RT_CONSUMER_BACKEND_THEAD)
  constexpr auto kExpectedDeviceType = infini::rt::Device::Type::kThead;
  constexpr bool kExpectAsyncMemcpySuccess = true;
#elif defined(INFINI_RT_CONSUMER_BACKEND_METAX)
  constexpr auto kExpectedDeviceType = infini::rt::Device::Type::kMetax;
  constexpr bool kExpectAsyncMemcpySuccess = true;
#elif defined(INFINI_RT_CONSUMER_BACKEND_MARS)
  constexpr auto kExpectedDeviceType = infini::rt::Device::Type::kMars;
  constexpr bool kExpectAsyncMemcpySuccess = true;
#elif defined(INFINI_RT_CONSUMER_BACKEND_MOORE)
  constexpr auto kExpectedDeviceType = infini::rt::Device::Type::kMoore;
  constexpr bool kExpectAsyncMemcpySuccess = true;
#elif defined(INFINI_RT_CONSUMER_BACKEND_CAMBRICON)
  constexpr auto kExpectedDeviceType = infini::rt::Device::Type::kCambricon;
  constexpr bool kExpectAsyncMemcpySuccess = true;
#elif defined(INFINI_RT_CONSUMER_BACKEND_ASCEND)
  constexpr auto kExpectedDeviceType = infini::rt::Device::Type::kAscend;
  constexpr bool kExpectAsyncMemcpySuccess = true;
#endif
  infini::rt::set_runtime_device_type(kExpectedDeviceType);
  if (infini::rt::runtime_device_type() != kExpectedDeviceType) {
    return 1;
  }

  std::array<std::uint8_t, 4> input{1, 2, 3, 4};
  std::array<std::uint8_t, 4> output{};
  void* ptr = nullptr;
  int device_count = 0;
  if (runtime::GetDeviceCount(&device_count) != runtime::kSuccess ||
      device_count <= 0) {
    return 0;
  }
  if (runtime::SetDevice(0) != runtime::kSuccess) {
    return 0;
  }
  int current_device = -1;
  if (runtime::GetDevice(&current_device) != runtime::kSuccess) {
    return 1;
  }
  if (current_device != 0) {
    return 1;
  }
  if (runtime::Malloc(&ptr, input.size()) != runtime::kSuccess) {
    return 1;
  }
  if (ptr == nullptr) {
    return 1;
  }
  if (runtime::Memcpy(ptr, input.data(), input.size(),
                      runtime::kMemcpyHostToDevice) != runtime::kSuccess) {
    return 1;
  }
  runtime::Stream stream{};
  const auto async_status = runtime::MemcpyAsync(
      ptr, input.data(), input.size(), runtime::kMemcpyHostToDevice, stream);
  if (kExpectAsyncMemcpySuccess && async_status != runtime::kSuccess) {
    return 1;
  }
  if (!kExpectAsyncMemcpySuccess && async_status == runtime::kSuccess) {
    return 1;
  }
  if (runtime::DeviceSynchronize() != runtime::kSuccess) {
    return 1;
  }
  if (runtime::Memcpy(output.data(), ptr, output.size(),
                      runtime::kMemcpyDeviceToHost) != runtime::kSuccess) {
    return 1;
  }
  if (output != input) {
    return 1;
  }
  if (runtime::Free(ptr) != runtime::kSuccess) {
    return 1;
  }
#endif

  return 0;
}
