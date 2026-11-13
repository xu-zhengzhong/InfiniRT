#include "sparse_tensor_view.h"

#include <cstdint>
#include <limits>
#include <utility>

namespace infini::rt {
namespace {

using Code = ValidationCode;
using Component = ValidationComponent;
using Size = SparseTensorView::Size;

static_assert(tensor_view_detail::kInlineMetadataCapacity >= 2,
              "Sparse component metadata must fit inline.");

ValidationResult Failure(Code code, Component component,
                         std::optional<Size> position = std::nullopt) noexcept {
  return {code, component, position};
}

bool Multiply(Size left, Size right, Size& result) noexcept {
  if (right != 0 && left > std::numeric_limits<Size>::max() / right) {
    return false;
  }
  result = left * right;
  return true;
}

struct ElementStorage {
  Size size;
  Size alignment;
};

ElementStorage Storage(DataType dtype) noexcept {
  switch (dtype) {
    case DataType::kInt8:
      return {sizeof(std::int8_t), alignof(std::int8_t)};
    case DataType::kInt16:
      return {sizeof(std::int16_t), alignof(std::int16_t)};
    case DataType::kInt32:
      return {sizeof(std::int32_t), alignof(std::int32_t)};
    case DataType::kInt64:
      return {sizeof(std::int64_t), alignof(std::int64_t)};
    case DataType::kUInt8:
      return {sizeof(std::uint8_t), alignof(std::uint8_t)};
    case DataType::kUInt16:
      return {sizeof(std::uint16_t), alignof(std::uint16_t)};
    case DataType::kUInt32:
      return {sizeof(std::uint32_t), alignof(std::uint32_t)};
    case DataType::kUInt64:
      return {sizeof(std::uint64_t), alignof(std::uint64_t)};
    case DataType::kFloat16:
      return {sizeof(Float16), alignof(Float16)};
    case DataType::kBFloat16:
      return {sizeof(BFloat16), alignof(BFloat16)};
    case DataType::kFloat32:
      return {sizeof(float), alignof(float)};
    case DataType::kFloat64:
      return {sizeof(double), alignof(double)};
    case DataType::kBool:
      return {sizeof(bool), alignof(bool)};
  }
  return {0, 0};
}

ValidationResult ValidateComponent(const TensorView& tensor,
                                   Component component) noexcept {
  const auto shape = tensor.shape();
  const auto strides = tensor.strides();
  if (shape.size() != strides.size()) {
    return Failure(Code::kInvalidMetadata, component);
  }

  const auto storage = Storage(tensor.dtype());
  if (storage.size == 0) {
    return Failure(Code::kInvalidMetadata, component);
  }
  const auto& device = tensor.device();
  if (device.type() < Device::Type::kCpu ||
      device.type() >= Device::Type::kCount || device.index() < 0) {
    return Failure(Code::kInvalidMetadata, component);
  }

  bool empty = false;
  for (Size axis = 0; axis < shape.size(); ++axis) {
    if (strides[axis] < 0) {
      return Failure(Code::kUnsupportedCapability, component, axis);
    }
    empty = empty || shape[axis] == 0;
  }

  // Empty components have no addressable elements; all nonnegative strides
  // are contiguous. Singleton axes do not constrain a nonempty layout.
  Size elements = empty ? 0 : 1;
  if (!empty) {
    for (Size axis = shape.size(); axis != 0;) {
      --axis;
      if (shape[axis] > 1) {
        if (elements >
            static_cast<Size>(std::numeric_limits<TensorView::Stride>::max())) {
          return Failure(Code::kIntegerOverflow, component, axis);
        }
        if (static_cast<Size>(strides[axis]) != elements) {
          return Failure(Code::kUnsupportedCapability, component, axis);
        }
      }
      if (!Multiply(elements, shape[axis], elements)) {
        return Failure(Code::kIntegerOverflow, component);
      }
    }
  }

  Size bytes = 0;
  if (!Multiply(elements, storage.size, bytes)) {
    return Failure(Code::kIntegerOverflow, component);
  }
  if (bytes == 0) return {};
  if (tensor.data() == nullptr) {
    return Failure(Code::kNullData, component);
  }
  const auto address = reinterpret_cast<std::uintptr_t>(tensor.data());
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - address) {
    return Failure(Code::kIntegerOverflow, component);
  }
  if (address % storage.alignment != 0) {
    return Failure(Code::kInvalidMetadata, component);
  }
  return {};
}

ValidationResult Access(bool allowed, const TensorView& component,
                        Component name, const TensorView*& out) noexcept {
  out = allowed ? &component : nullptr;
  return allowed ? ValidationResult{} : Failure(Code::kInvalidState, name);
}

}  // namespace

SparseTensorView::SparseTensorView(TensorLayout layout, Shape shape, Size nnz,
                                   Size numel, bool coalesced,
                                   const TensorView& indices,
                                   const TensorView* col_indices,
                                   const TensorView& values) noexcept
    : layout_{layout},
      shape_{shape},
      nnz_{nnz},
      numel_{numel},
      coalesced_{coalesced},
      indices_{indices},
      values_{values} {
  if (col_indices != nullptr) col_indices_.emplace(*col_indices);
}

ValidationResult SparseTensorView::Create(
    TensorLayout layout, ShapeView shape,
    std::initializer_list<const TensorView*> components,
    std::optional<SparseTensorView>& out, bool coalesced) noexcept {
  const auto fail = [&](Code code, Component component,
                        std::optional<Size> position = std::nullopt) {
    out.reset();
    return Failure(code, component, position);
  };
  if (layout != TensorLayout::kCoo && layout != TensorLayout::kCsr) {
    return fail(Code::kUnsupportedLayout, Component::kLayout);
  }
  if (shape.size() != 2) {
    return fail(Code::kUnsupportedCapability, Component::kShape);
  }
  if (shape.data() == nullptr) {
    return fail(Code::kInvalidMetadata, Component::kShape);
  }
  const bool coo = layout == TensorLayout::kCoo;
  if (components.size() != (coo ? 2 : 3)) {
    return fail(Code::kInvalidMetadata, Component::kComponents);
  }
  for (const auto* component : components) {
    if (component == nullptr) {
      return fail(Code::kInvalidMetadata, Component::kComponents);
    }
  }
  if (!coo && coalesced) {
    return fail(Code::kInvalidMetadata, Component::kCoalesced);
  }

  const auto& indices = *components.begin()[0];
  const auto* col_indices = coo ? nullptr : components.begin()[1];
  const auto& values = *components.begin()[coo ? 1 : 2];
  const Component index_component =
      coo ? Component::kIndices : Component::kCrowIndices;
  const DataType index_dtype = indices.dtype();
  if ((coo && index_dtype != DataType::kInt64) ||
      (!coo && index_dtype != DataType::kInt32 &&
       index_dtype != DataType::kInt64)) {
    return fail(Code::kInvalidIndexType, index_component);
  }
  if (!coo && col_indices->dtype() != index_dtype) {
    return fail(Code::kInvalidIndexType, Component::kColIndices);
  }
  if (values.ndim() != 1) {
    return fail(values.ndim() > 1 ? Code::kUnsupportedCapability
                                  : Code::kInvalidMetadata,
                Component::kValues);
  }
  if (indices.ndim() != (coo ? 2 : 1)) {
    return fail(Code::kInvalidMetadata, index_component);
  }
  if (!coo && col_indices->ndim() != 1) {
    return fail(Code::kInvalidMetadata, Component::kColIndices);
  }

  const Size nnz = values.shape()[0];
  const auto index_max = index_dtype == DataType::kInt32
                             ? static_cast<std::uintmax_t>(
                                   std::numeric_limits<std::int32_t>::max())
                             : static_cast<std::uintmax_t>(
                                   std::numeric_limits<std::int64_t>::max());
  for (Size axis = 0; axis < 2; ++axis) {
    if (shape[axis] > index_max) {
      return fail(Code::kIntegerOverflow, Component::kShape, axis);
    }
  }
  if (nnz > index_max) {
    return fail(Code::kIntegerOverflow, Component::kValues);
  }
  Size numel = 0;
  if (!Multiply(shape[0], shape[1], numel)) {
    return fail(Code::kIntegerOverflow, Component::kShape);
  }
  if (numel == 0 && nnz != 0) {
    return fail(Code::kInvalidMetadata, Component::kValues);
  }
  if (coo) {
    if (indices.shape()[0] != 2 || indices.shape()[1] != nnz) {
      return fail(Code::kInvalidMetadata, Component::kIndices);
    }
  } else {
    if (shape[0] == std::numeric_limits<Size>::max()) {
      return fail(Code::kIntegerOverflow, Component::kShape, 0);
    }
    if (indices.shape()[0] != shape[0] + 1) {
      return fail(Code::kInvalidMetadata, Component::kCrowIndices);
    }
    if (col_indices->shape()[0] != nnz) {
      return fail(Code::kInvalidMetadata, Component::kColIndices);
    }
  }

  const Component names[] = {index_component,
                             coo ? Component::kValues : Component::kColIndices,
                             Component::kValues};
  Size index = 0;
  for (const auto* component : components) {
    const Component name = names[index++];
    if (component->device() != values.device()) {
      return fail(Code::kDeviceMismatch, name);
    }
    const auto status = ValidateComponent(*component, name);
    if (!status) return fail(status.code, status.component, status.position);
  }

  // Materialize before replacing `out`: input metadata may belong to its old
  // descriptor. All validated component ranks fit TensorView's inline storage.
  SparseTensorView view{layout,  {shape[0], shape[1]}, nnz,   numel, coalesced,
                        indices, col_indices,          values};
  out.emplace(std::move(view));
  return {};
}

ValidationResult SparseTensorView::is_coalesced(bool& out) const noexcept {
  out = layout_ == TensorLayout::kCoo && coalesced_;
  return layout_ == TensorLayout::kCoo
             ? ValidationResult{}
             : Failure(Code::kInvalidState, Component::kCoalesced);
}

ValidationResult SparseTensorView::indices(
    const TensorView*& out) const& noexcept {
  return Access(layout_ == TensorLayout::kCoo && coalesced_, indices_,
                Component::kIndices, out);
}

ValidationResult SparseTensorView::raw_indices(
    const TensorView*& out) const& noexcept {
  return Access(layout_ == TensorLayout::kCoo, indices_, Component::kIndices,
                out);
}

ValidationResult SparseTensorView::crow_indices(
    const TensorView*& out) const& noexcept {
  return Access(layout_ == TensorLayout::kCsr, indices_,
                Component::kCrowIndices, out);
}

ValidationResult SparseTensorView::col_indices(
    const TensorView*& out) const& noexcept {
  out = nullptr;
  if (layout_ != TensorLayout::kCsr) {
    return Failure(Code::kInvalidState, Component::kColIndices);
  }
  out = &*col_indices_;
  return {};
}

ValidationResult SparseTensorView::values(
    const TensorView*& out) const& noexcept {
  return Access(layout_ == TensorLayout::kCsr || coalesced_, values_,
                Component::kValues, out);
}

ValidationResult SparseTensorView::raw_values(
    const TensorView*& out) const& noexcept {
  out = &values_;
  return {};
}

}  // namespace infini::rt
