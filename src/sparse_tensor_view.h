#ifndef INFINI_RT_SPARSE_TENSOR_VIEW_H_
#define INFINI_RT_SPARSE_TENSOR_VIEW_H_

#include <array>
#include <cstddef>
#include <initializer_list>
#include <optional>

#include "tensor_layout.h"
#include "tensor_view.h"

namespace infini::rt {

// These codes describe metadata only. Content and execution checks belong
// to Ops.
enum class ValidationCode {
  kSuccess,
  kInvalidMetadata,
  kUnsupportedLayout,
  kUnsupportedCapability,
  kInvalidIndexType,
  kDeviceMismatch,
  kNullData,
  kIntegerOverflow,
  kInvalidState,
  kAllocationFailure,
};

enum class ValidationComponent {
  kNone,
  kLayout,
  kShape,
  kComponents,
  kIndices,
  kCrowIndices,
  kColIndices,
  kValues,
  kCoalesced,
};

struct [[nodiscard]] ValidationResult {
  ValidationCode code{ValidationCode::kSuccess};

  ValidationComponent component{ValidationComponent::kNone};

  std::optional<std::size_t> position;

  constexpr bool ok() const noexcept {
    return code == ValidationCode::kSuccess;
  }

  constexpr explicit operator bool() const noexcept { return ok(); }
};

// Owns metadata and borrows data. Successful construction does not validate
// index contents or establish provider support. Copies do not extend data life.
class SparseTensorView {
 public:
  using Size = TensorView::Size;

  using Shape = std::array<Size, 2>;

  using ShapeView = TensorView::ShapeView;

  // Components are ordered as {indices, values} for COO and
  // {crow_indices, col_indices, values} for CSR. On failure, `out` is empty.
  // Shape and component metadata are copied only after all checks succeed.
  // Supported metadata fits inline; these factories perform no allocations,
  // data reads, backend calls, or stream synchronization.
  static ValidationResult Create(
      TensorLayout layout, ShapeView shape,
      std::initializer_list<const TensorView*> components,
      std::optional<SparseTensorView>& out, bool coalesced = false) noexcept;

  static ValidationResult Create(
      TensorLayout layout, std::initializer_list<Size> shape,
      std::initializer_list<const TensorView*> components,
      std::optional<SparseTensorView>& out, bool coalesced = false) noexcept {
    return Create(layout, ShapeView{shape.begin(), shape.size()}, components,
                  out, coalesced);
  }

  static ValidationResult Create(
      TensorLayout layout, const Shape& shape,
      std::initializer_list<const TensorView*> components,
      std::optional<SparseTensorView>& out, bool coalesced = false) noexcept {
    return Create(layout, ShapeView{shape.data(), shape.size()}, components,
                  out, coalesced);
  }

  static ValidationResult CreateCoo(ShapeView shape, const TensorView& indices,
                                    const TensorView& values,
                                    std::optional<SparseTensorView>& out,
                                    bool coalesced = false) noexcept {
    return Create(TensorLayout::kCoo, shape, {&indices, &values}, out,
                  coalesced);
  }

  static ValidationResult CreateCoo(std::initializer_list<Size> shape,
                                    const TensorView& indices,
                                    const TensorView& values,
                                    std::optional<SparseTensorView>& out,
                                    bool coalesced = false) noexcept {
    return CreateCoo(ShapeView{shape.begin(), shape.size()}, indices, values,
                     out, coalesced);
  }

  static ValidationResult CreateCoo(const Shape& shape,
                                    const TensorView& indices,
                                    const TensorView& values,
                                    std::optional<SparseTensorView>& out,
                                    bool coalesced = false) noexcept {
    return CreateCoo(ShapeView{shape.data(), shape.size()}, indices, values,
                     out, coalesced);
  }

  static ValidationResult CreateCsr(
      ShapeView shape, const TensorView& crow_indices,
      const TensorView& col_indices, const TensorView& values,
      std::optional<SparseTensorView>& out) noexcept {
    return Create(TensorLayout::kCsr, shape,
                  {&crow_indices, &col_indices, &values}, out);
  }

  static ValidationResult CreateCsr(
      std::initializer_list<Size> shape, const TensorView& crow_indices,
      const TensorView& col_indices, const TensorView& values,
      std::optional<SparseTensorView>& out) noexcept {
    return CreateCsr(ShapeView{shape.begin(), shape.size()}, crow_indices,
                     col_indices, values, out);
  }

  static ValidationResult CreateCsr(
      const Shape& shape, const TensorView& crow_indices,
      const TensorView& col_indices, const TensorView& values,
      std::optional<SparseTensorView>& out) noexcept {
    return CreateCsr(ShapeView{shape.data(), shape.size()}, crow_indices,
                     col_indices, values, out);
  }

  SparseTensorView(const SparseTensorView&) noexcept = default;

  // Keep moved-from descriptors valid; all metadata is small and inline.
  SparseTensorView(SparseTensorView&& other) noexcept
      : SparseTensorView(static_cast<const SparseTensorView&>(other)) {}

  SparseTensorView& operator=(const SparseTensorView&) = delete;

  SparseTensorView& operator=(SparseTensorView&&) = delete;

  TensorLayout layout() const noexcept { return layout_; }

  Shape shape() const noexcept { return shape_; }

  Size ndim() const noexcept { return 2; }

  Size sparse_dim() const noexcept { return 2; }

  Size dense_dim() const noexcept { return 0; }

  Size batch_dim() const noexcept { return 0; }

  Size nnz() const noexcept { return nnz_; }

  Size numel() const noexcept { return numel_; }

  DataType dtype() const noexcept { return values_.dtype(); }

  DataType index_dtype() const noexcept { return indices_.dtype(); }

  Device device() const noexcept { return values_.device(); }

  // A COO declaration is not proof of sorted, unique coordinates.
  ValidationResult is_coalesced(bool& out) const noexcept;

  // Accessors clear `out` on failure. Returned pointers borrow this descriptor.
  // Raw access bypasses only the COO coalesced-state requirement.
  ValidationResult indices(const TensorView*& out) const& noexcept;

  ValidationResult raw_indices(const TensorView*& out) const& noexcept;

  ValidationResult crow_indices(const TensorView*& out) const& noexcept;

  ValidationResult col_indices(const TensorView*& out) const& noexcept;

  ValidationResult values(const TensorView*& out) const& noexcept;

  ValidationResult raw_values(const TensorView*& out) const& noexcept;

  ValidationResult indices(const TensorView*&) const&& = delete;

  ValidationResult raw_indices(const TensorView*&) const&& = delete;

  ValidationResult crow_indices(const TensorView*&) const&& = delete;

  ValidationResult col_indices(const TensorView*&) const&& = delete;

  ValidationResult values(const TensorView*&) const&& = delete;

  ValidationResult raw_values(const TensorView*&) const&& = delete;

 private:
  SparseTensorView(TensorLayout layout, Shape shape, Size nnz, Size numel,
                   bool coalesced, const TensorView& indices,
                   const TensorView* col_indices,
                   const TensorView& values) noexcept;

  TensorLayout layout_;

  Shape shape_;

  Size nnz_;

  Size numel_;

  bool coalesced_;

  TensorView indices_;

  std::optional<TensorView> col_indices_;

  TensorView values_;
};

}  // namespace infini::rt

#endif
