#include <infini/rt.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

thread_local bool count_allocations = false;
thread_local std::size_t allocation_count = 0;
int failures = 0;
int checks = 0;

void Check(bool condition, const char* label) {
  ++checks;
  if (!condition) {
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", label);
  }
}
}  // namespace

void* operator new(std::size_t size) {
  if (count_allocations) ++allocation_count;
  if (void* pointer = std::malloc(size == 0 ? 1 : size)) return pointer;
  throw std::bad_alloc{};
}

void* operator new[](std::size_t size) { return ::operator new(size); }

void operator delete(void* pointer) noexcept { std::free(pointer); }

void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }

void operator delete(void* pointer, std::size_t) noexcept {
  ::operator delete(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept {
  ::operator delete(pointer);
}

namespace {

using namespace infini::rt;
using Code = ValidationCode;
using Optional = std::optional<SparseTensorView>;
const Device cpu{Device::Type::kCpu};

void Expect(ValidationResult result, Code code, const Optional& out,
            const char* label) {
  Check(result.code == code, label);
  Check(out.has_value() == (code == Code::kSuccess),
        "Factory publishes iff success.");
  Check(result.ok() == (code == Code::kSuccess), "Status ok agrees with code.");
  Check(static_cast<bool>(result) == result.ok(),
        "Explicit bool agrees with status.");
  if (result.ok())
    Check(result.component == ValidationComponent::kNone,
          "Success has no error component.");
  if (result.ok())
    Check(!result.position.has_value(), "Success has no error position.");
}

using Component = ValidationComponent;

void ExpectFailure(ValidationResult result, Code code, Component component,
                   const Optional& out, const char* label,
                   std::optional<std::size_t> position = std::nullopt) {
  Expect(result, code, out, label);
  Check(result.component == component, "Failure identifies the component.");
  Check(result.position == position, "Failure identifies the metadata axis.");
}

void TestBasics(void* memory) {
  TensorView indices{memory, {2, 3}, DataType::kInt64, cpu, {3, 1}};
  TensorView values{memory, {3}, DataType::kFloat32, cpu, {1}};
  Optional out;
  Expect(SparseTensorView::CreateCoo({4, 5}, indices, values, out),
         Code::kSuccess, out,
         "COO accepts unreadable components without reading contents.");
  if (!out) return;
  Check(out->layout() == TensorLayout::kCoo &&
            out->shape() == SparseTensorView::Shape{4, 5},
        "COO shape and layout.");
  Check(out->nnz() == 3 && out->numel() == 20 && out->ndim() == 2 &&
            out->sparse_dim() == 2 && out->dense_dim() == 0 &&
            out->batch_dim() == 0,
        "COO dimension metadata.");
  Check(out->dtype() == DataType::kFloat32 &&
            out->index_dtype() == DataType::kInt64 && out->device() == cpu,
        "COO dtype and full device.");
  bool coalesced = true;
  Check(out->is_coalesced(coalesced).ok() && !coalesced,
        "Omitted coalesced is false.");
  const TensorView* component = &indices;
  Check(out->indices(component).code == Code::kInvalidState && !component,
        "Uncoalesced indices access clears output.");
  component = &values;
  Check(out->values(component).code == Code::kInvalidState && !component,
        "Uncoalesced values access clears output.");
  Check(out->raw_indices(component).ok() && component->data() == memory,
        "Raw COO indices access.");
  Check(out->raw_values(component).ok() && component->data() == memory,
        "Raw COO values access.");
  Check(out->crow_indices(component).code == Code::kInvalidState && !component,
        "COO rejects CSR crow accessor.");
  Check(out->col_indices(component).code == Code::kInvalidState && !component,
        "COO rejects CSR col accessor.");
  Expect(SparseTensorView::CreateCoo({4, 5}, indices, values, out, true),
         Code::kSuccess, out,
         "Coalesced declaration is accepted without content verification.");
  if (!out) return;
  Check(out->is_coalesced(coalesced).ok() && coalesced,
        "Explicit coalesced retained.");
  Check(out->indices(component).ok() && component->shape() == indices.shape(),
        "Coalesced indices accessible.");
  Check(out->values(component).ok() && component->shape() == values.shape(),
        "Coalesced values accessible.");
  for (DataType dtype :
       {DataType::kInt8, DataType::kInt16, DataType::kInt32, DataType::kInt64,
        DataType::kUInt8, DataType::kUInt16, DataType::kUInt32,
        DataType::kUInt64, DataType::kFloat16, DataType::kBFloat16,
        DataType::kFloat32, DataType::kFloat64, DataType::kBool}) {
    TensorView typed_values{memory, {3}, dtype, cpu, {1}};
    Expect(SparseTensorView::CreateCoo({4, 5}, indices, typed_values, out),
           Code::kSuccess, out,
           "Every existing value storage dtype is representable.");
    if (!out) return;
  }
  for (DataType dtype : {DataType::kInt32, DataType::kInt64}) {
    TensorView crow{memory, {5}, dtype, cpu, {1}};
    TensorView col{memory, {3}, dtype, cpu, {1}};
    Expect(SparseTensorView::CreateCsr({4, 5}, crow, col, values, out),
           Code::kSuccess, out, "CSR accepts both signed index widths.");
    if (!out) return;
    Check(
        out->crow_indices(component).ok() && component->shape() == crow.shape(),
        "CSR crow accessor.");
    Check(out->col_indices(component).ok() && component->shape() == col.shape(),
          "CSR col accessor.");
    Check(out->values(component).ok(), "CSR values require no coalesced flag.");
    Check(out->indices(component).code == Code::kInvalidState && !component,
          "CSR rejects COO indices accessor.");
    Check(out->raw_indices(component).code == Code::kInvalidState && !component,
          "CSR rejects raw COO indices accessor.");
    Check(out->is_coalesced(coalesced).code == Code::kInvalidState,
          "CSR rejects COO coalesced query.");
  }
}

void TestEmptyAndSingleton(void* memory) {
  TensorView indices{nullptr, {2, 0}, DataType::kInt64, cpu, {0, 1}};
  TensorView indices_alt{nullptr, {2, 0}, DataType::kInt64, cpu, {1, 1}};
  TensorView values{nullptr, {0}, DataType::kFloat32, cpu, {1}};
  Optional out;
  for (auto shape :
       {SparseTensorView::Shape{0, 0}, SparseTensorView::Shape{0, 8},
        SparseTensorView::Shape{8, 0}, SparseTensorView::Shape{4, 8}}) {
    SparseTensorView::ShapeView shape_view{shape.data(), shape.size()};
    Expect(SparseTensorView::CreateCoo(shape_view, indices, values, out),
           Code::kSuccess, out,
           "Empty COO handles zero logical axes and zero nnz.");
    if (!out) return;
    bool coalesced = true;
    Check(out->nnz() == 0 && out->is_coalesced(coalesced).ok() && !coalesced,
          "Empty COO keeps false declaration.");
    Expect(
        SparseTensorView::CreateCoo(shape_view, indices_alt, values, out, true),
        Code::kSuccess, out, "Alternate empty continuous strides accepted.");
    if (!out) return;
    Check(out->is_coalesced(coalesced).ok() && coalesced,
          "Empty COO preserves true declaration.");
    TensorView crow{memory, {shape[0] + 1}, DataType::kInt32, cpu, {1}};
    TensorView col{nullptr, {0}, DataType::kInt32, cpu, {1}};
    Expect(SparseTensorView::CreateCsr(shape_view, crow, col, values, out),
           Code::kSuccess, out, "Empty CSR still has crow component.");
    if (!out) return;
  }
  TensorView null_crow{nullptr, {1}, DataType::kInt32, cpu, {1}};
  TensorView col{nullptr, {0}, DataType::kInt32, cpu, {1}};
  Expect(SparseTensorView::CreateCsr({0, 8}, null_crow, col, values, out),
         Code::kNullData, out, "Zero-row CSR still requires nonnull crow.");
  TensorView empty_negative{nullptr, {0}, DataType::kFloat32, cpu, {-1}};
  Expect(SparseTensorView::CreateCoo({0, 8}, indices, empty_negative, out),
         Code::kUnsupportedCapability, out, "Empty negative stride rejected.");
  TensorView one_index{memory, {2, 1}, DataType::kInt64, cpu, {1, 29}};
  TensorView one_value{memory, {1}, DataType::kFloat32, cpu, {37}};
  Expect(SparseTensorView::CreateCoo({4, 5}, one_index, one_value, out),
         Code::kSuccess, out,
         "Singleton strides do not imply noncontiguous storage.");
  if (!out) return;
  Expect(SparseTensorView::CreateCoo({0, 5}, one_index, one_value, out),
         Code::kInvalidMetadata, out,
         "Zero logical extent rejects nonzero nnz.");
  TensorView empty_misaligned{
      reinterpret_cast<void*>(1), {0}, DataType::kFloat32, cpu, {1}};
  Expect(SparseTensorView::CreateCoo({0, 8}, indices, empty_misaligned, out),
         Code::kSuccess, out,
         "Empty unused component needs no address alignment.");
  if (!out) return;
}

void TestOverflow(void* memory) {
  constexpr auto max_size = std::numeric_limits<std::size_t>::max();
  constexpr auto max_i64 =
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
  constexpr auto max_i32 =
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
  TensorView indices{nullptr, {2, 0}, DataType::kInt64, cpu, {0, 1}};
  TensorView values{nullptr, {0}, DataType::kFloat64, cpu, {1}};
  Optional out;
  if constexpr (std::numeric_limits<std::size_t>::digits >= 64) {
    Expect(SparseTensorView::CreateCoo({max_i64, 2}, indices, values, out),
           Code::kSuccess, out,
           "Sparse logical numel is not multiplied by value element size.");
    if (!out) return;
    Expect(SparseTensorView::CreateCoo({max_i64, 3}, indices, values, out),
           Code::kIntegerOverflow, out, "Logical numel overflow rejected.");
    Expect(SparseTensorView::CreateCoo({max_i64 + 1, 0}, indices, values, out),
           Code::kIntegerOverflow, out,
           "Logical extent index-width overflow rejected.");
  }
  TensorView crow32{memory, {1}, DataType::kInt32, cpu, {1}};
  TensorView col32{nullptr, {0}, DataType::kInt32, cpu, {1}};
  Expect(SparseTensorView::CreateCsr({0, max_i32}, crow32, col32, values, out),
         Code::kSuccess, out, "CSR int32 boundary extent accepted.");
  if (!out) return;
  Expect(
      SparseTensorView::CreateCsr({0, max_i32 + 1}, crow32, col32, values, out),
      Code::kIntegerOverflow, out, "CSR int32 extent overflow rejected.");
  if constexpr (std::numeric_limits<std::size_t>::digits >= 64) {
    TensorView crow_huge{memory, {max_i64 + 1}, DataType::kInt64, cpu, {1}};
    TensorView col64{nullptr, {0}, DataType::kInt64, cpu, {1}};
    Expect(SparseTensorView::CreateCsr({max_i64, 0}, crow_huge, col64, values,
                                       out),
           Code::kIntegerOverflow, out, "CSR crow byte overflow rejected.");
  }
  const std::size_t huge_nnz = max_size / 16 + 1;
  TensorView huge_indices{memory,
                          {2, huge_nnz},
                          DataType::kInt64,
                          cpu,
                          {static_cast<std::ptrdiff_t>(huge_nnz), 1}};
  TensorView huge_values{memory, {huge_nnz}, DataType::kUInt8, cpu, {1}};
  Expect(SparseTensorView::CreateCoo({1, 1}, huge_indices, huge_values, out),
         Code::kIntegerOverflow, out, "COO component byte overflow rejected.");
  TensorView end_indices{
      reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 7),
      {2, 1},
      DataType::kInt64,
      cpu,
      {1, 1}};
  TensorView one_value{memory, {1}, DataType::kFloat32, cpu, {1}};
  Expect(SparseTensorView::CreateCoo({1, 1}, end_indices, one_value, out),
         Code::kIntegerOverflow, out,
         "Component address endpoint overflow rejected.");
  TensorView long_stride{memory,
                         {2, 3},
                         DataType::kInt64,
                         cpu,
                         {std::numeric_limits<std::ptrdiff_t>::max(), 1}};
  TensorView three_values{memory, {3}, DataType::kFloat32, cpu, {1}};
  auto result =
      SparseTensorView::CreateCoo({4, 5}, long_stride, three_values, out);
  Check(!result.ok() && !out,
        "Extreme noncontiguous stride rejected without arithmetic overflow.");
  if constexpr (std::numeric_limits<std::size_t>::digits >= 64) {
    // The boundary is checked using metadata, without allocating index arrays.
    TensorView crow{memory, {2}, DataType::kInt32, cpu, {1}};
    TensorView max_col{memory, {max_i32}, DataType::kInt32, cpu, {1}};
    TensorView max_values{memory, {max_i32}, DataType::kFloat64, cpu, {1}};
    Expect(SparseTensorView::CreateCsr({1, max_i32}, crow, max_col, max_values,
                                       out),
           Code::kSuccess, out, "CSR int32 maximum nnz is representable.");
    if (!out) return;
    TensorView excessive_col{memory, {max_i32 + 1}, DataType::kInt32, cpu, {1}};
    TensorView excessive_values{
        memory, {max_i32 + 1}, DataType::kFloat64, cpu, {1}};
    ExpectFailure(
        SparseTensorView::CreateCsr({1, max_i32}, crow, excessive_col,
                                    excessive_values, out),
        Code::kIntegerOverflow, Component::kValues, out,
        "CSR int32 nnz overflow is rejected before storage inspection.");
    TensorView excessive_coo_values{
        memory, {max_i64 + 1}, DataType::kUInt8, cpu, {1}};
    ExpectFailure(
        SparseTensorView::CreateCoo({1, 1}, indices, excessive_coo_values, out),
        Code::kIntegerOverflow, Component::kValues, out,
        "COO nnz beyond int64 is rejected before component geometry.");
    ExpectFailure(
        SparseTensorView::CreateCoo({0, max_i64 + 1}, indices, values, out),
        Code::kIntegerOverflow, Component::kShape, out,
        "An index-width failure reports the logical shape axis.", 1);
  }
}

void TestMetadataOwnershipAndAllocations(void* memory) {
  Optional out;
  {
    std::array<std::size_t, 2> logical_shape{4, 5};
    TensorView indices{memory, {2, 3}, DataType::kInt64, cpu, {3, 1}};
    TensorView values{memory, {3}, DataType::kFloat32, cpu, {1}};
    allocation_count = 0;
    count_allocations = true;
    const auto result = SparseTensorView::CreateCoo(
        SparseTensorView::ShapeView{logical_shape.data(), logical_shape.size()},
        indices, values, out);
    if (out) {
      SparseTensorView copied{*out};
      SparseTensorView moved{std::move(copied)};
      const TensorView* component = nullptr;
      const auto access = moved.raw_indices(component);
      Check(access.ok() && component->data() == memory,
            "Copy and move preserve component.");
      Check(copied.raw_indices(component).ok() && component->data() == memory &&
                component->shape().size() == 2 && component->shape()[1] == 3 &&
                copied.shape() == SparseTensorView::Shape{4, 5} &&
                copied.nnz() == 3,
            "Moved-from descriptor retains valid metadata.");
    }
    count_allocations = false;
    Check(result.ok() && allocation_count == 0,
          "Factory, copy, move and access perform no allocations.");
    logical_shape[0] = 999;
    indices.data() = nullptr;
    values.data() = nullptr;
  }
  if (!out) return;
  const TensorView* component = nullptr;
  Check(out->shape() == SparseTensorView::Shape{4, 5},
        "Logical shape is owned.");
  Check(out->raw_indices(component).ok() && component->data() == memory &&
            component->shape()[0] == 2 && component->shape()[1] == 3,
        "Component metadata survives original view destruction and mutation.");
  Check(out->raw_values(component).ok() && component->data() == memory,
        "Values pointer is copied, data remains borrowed.");
  const TensorView* old_indices = nullptr;
  const TensorView* old_values = nullptr;
  Check(out->raw_indices(old_indices).ok() && out->raw_values(old_values).ok(),
        "Read self-replacement source components.");
  const SparseTensorView::Shape copied_shape = out->shape();
  Expect(SparseTensorView::CreateCoo(copied_shape, *old_indices, *old_values,
                                     out, true),
         Code::kSuccess, out,
         "Shape variable and self-replacement factory are safe.");
  if (!out) return;
  Check(
      out->indices(old_indices).ok() && out->values(old_values).ok() &&
          old_indices->data() == memory && old_values->data() == memory,
      "Self-replacement retains borrowed components and updates declaration.");
  Expect(
      SparseTensorView::CreateCoo(out->shape(), *old_indices, *old_values, out),
      Code::kSuccess, out,
      "Temporary shape result and self-replacement are safe.");
  if (!out) return;
  bool coalesced = true;
  Check(out->is_coalesced(coalesced).ok() && !coalesced,
        "Self-replacement keeps the requested coalesced state.");
  TensorView bad{memory, {2, 3}, DataType::kInt32, cpu, {3, 1}};
  TensorView values{memory, {3}, DataType::kFloat32, cpu, {1}};
  count_allocations = true;
  allocation_count = 0;
  const auto failure = SparseTensorView::CreateCoo({4, 5}, bad, values, out);
  count_allocations = false;
  Check(!failure.ok() && !out && allocation_count == 0,
        "Failure clears result without allocating.");
  TensorView crow{memory, {5}, DataType::kInt64, cpu, {1}};
  TensorView col{memory, {3}, DataType::kInt64, cpu, {1}};
  allocation_count = 0;
  count_allocations = true;
  const auto csr_status =
      SparseTensorView::CreateCsr({4, 5}, crow, col, values, out);
  if (out) {
    SparseTensorView copied{*out};
    SparseTensorView moved{std::move(copied)};
    const TensorView* old_crow = nullptr;
    const TensorView* old_col = nullptr;
    const TensorView* csr_values = nullptr;
    Check(moved.crow_indices(old_crow).ok() &&
              moved.col_indices(old_col).ok() && moved.values(csr_values).ok(),
          "CSR copy and move retain all component metadata.");
    Check(copied.col_indices(component).ok() && component->data() == memory,
          "Moved-from CSR retains the optional column component.");
    Check(out->crow_indices(old_crow).ok() && out->col_indices(old_col).ok() &&
              out->values(csr_values).ok(),
          "CSR self-replacement source is accessible.");
    Check(SparseTensorView::CreateCsr(out->shape(), *old_crow, *old_col,
                                      *csr_values, out)
              .ok(),
          "CSR self-replacement preserves its three borrowed components.");
  }
  count_allocations = false;
  Check(csr_status.ok() && allocation_count == 0,
        "CSR factory, copy, move and self-replacement perform no allocations.");
}

void TestDetailedFailures(void* memory) {
  TensorView indices{memory, {2, 3}, DataType::kInt64, cpu, {3, 1}};
  TensorView values{memory, {3}, DataType::kFloat32, cpu, {1}};
  TensorView crow{memory, {5}, DataType::kInt64, cpu, {1}};
  TensorView col{memory, {3}, DataType::kInt64, cpu, {1}};
  Optional out;
  const auto seed = [&] {
    Check(SparseTensorView::CreateCoo({4, 5}, indices, values, out).ok(),
          "Seed a successful result before testing failure replacement.");
  };
  const auto reject = [&](TensorLayout layout,
                          std::initializer_list<SparseTensorView::Size> shape,
                          std::initializer_list<const TensorView*> components,
                          Code code, Component component, const char* label,
                          bool coalesced = false) {
    seed();
    ExpectFailure(
        SparseTensorView::Create(layout, shape, components, out, coalesced),
        code, component, out, label);
  };
  for (auto layout : {TensorLayout::kCsc, TensorLayout::kBsr,
                      TensorLayout::kBsc, static_cast<TensorLayout>(999)}) {
    reject(layout, {4, 5}, {&indices, &values}, Code::kUnsupportedLayout,
           Component::kLayout, "Unsupported layout is rejected.");
  }
  for (auto shape : {std::initializer_list<std::size_t>{}, {4}, {4, 5, 6}}) {
    reject(TensorLayout::kCoo, shape, {&indices, &values},
           Code::kUnsupportedCapability, Component::kShape,
           "Only explicit rank two is supported.");
  }
  seed();
  ExpectFailure(
      SparseTensorView::CreateCoo(SparseTensorView::ShapeView{nullptr, 2},
                                  indices, values, out),
      Code::kInvalidMetadata, Component::kShape, out,
      "Null shape metadata is rejected before reading it.");
  for (auto layout : {TensorLayout::kCoo, TensorLayout::kCsr}) {
    reject(layout, {4, 5}, {}, Code::kInvalidMetadata, Component::kComponents,
           "Missing components are rejected.");
    reject(layout, {4, 5}, {&indices, &crow, &col, &values},
           Code::kInvalidMetadata, Component::kComponents,
           "Extra components are rejected.");
  }
  reject(TensorLayout::kCoo, {4, 5}, {nullptr, &values}, Code::kInvalidMetadata,
         Component::kComponents, "Null component object is rejected.");
  reject(TensorLayout::kCsr, {4, 5}, {&crow, nullptr, &values},
         Code::kInvalidMetadata, Component::kComponents,
         "Null CSR component object is rejected.");
  reject(TensorLayout::kCsr, {4, 5}, {&crow, &col, &values},
         Code::kInvalidMetadata, Component::kCoalesced,
         "CSR rejects a COO coalesced declaration.", true);
  for (auto dtype : {DataType::kInt32, DataType::kUInt64, DataType::kFloat64}) {
    TensorView wrong{memory, {2, 3}, dtype, cpu, {3, 1}};
    reject(TensorLayout::kCoo, {4, 5}, {&wrong, &values},
           Code::kInvalidIndexType, Component::kIndices,
           "COO accepts only signed int64 indices.");
  }
  TensorView col32{memory, {3}, DataType::kInt32, cpu, {1}};
  reject(TensorLayout::kCsr, {4, 5}, {&crow, &col32, &values},
         Code::kInvalidIndexType, Component::kColIndices,
         "CSR index widths must agree.");
  TensorView crow_unsigned{memory, {5}, DataType::kUInt64, cpu, {1}};
  reject(TensorLayout::kCsr, {4, 5}, {&crow_unsigned, &col, &values},
         Code::kInvalidIndexType, Component::kCrowIndices,
         "CSR rejects unsigned indices.");
  TensorView hybrid{memory, {3, 1}, DataType::kFloat32, cpu, {1, 1}};
  TensorView scalar{memory, {}, DataType::kFloat32, cpu, {}};
  reject(TensorLayout::kCoo, {4, 5}, {&indices, &hybrid},
         Code::kUnsupportedCapability, Component::kValues,
         "Hybrid values are unsupported.");
  reject(TensorLayout::kCoo, {4, 5}, {&indices, &scalar},
         Code::kInvalidMetadata, Component::kValues,
         "Scalar values have invalid rank.");
  TensorView invalid_dtype{memory, {3}, static_cast<DataType>(127), cpu, {1}};
  reject(TensorLayout::kCoo, {4, 5}, {&indices, &invalid_dtype},
         Code::kInvalidMetadata, Component::kValues,
         "Unknown value dtype returns a checked failure.");

  const TensorView* originals[] = {&indices, &crow, &col, &values};
  const Component names[] = {Component::kIndices, Component::kCrowIndices,
                             Component::kColIndices, Component::kValues};
  for (std::size_t role = 0; role < 4; ++role) {
    const auto& original = *originals[role];
    const auto reject_component = [&](const TensorView& replacement, Code code,
                                      const char* label,
                                      std::optional<std::size_t> axis =
                                          std::nullopt) {
      seed();
      const auto result = role == 0 || role == 3
                              ? SparseTensorView::CreateCoo(
                                    {4, 5}, role == 0 ? replacement : indices,
                                    role == 3 ? replacement : values, out)
                              : SparseTensorView::CreateCsr(
                                    {4, 5}, role == 1 ? replacement : crow,
                                    role == 2 ? replacement : col, values, out);
      ExpectFailure(result, code, names[role], out, label, axis);
    };
    TensorView null{nullptr, original.shape(), original.dtype(), cpu,
                    original.strides()};
    reject_component(null, Code::kNullData,
                     "Nonempty components require data.");
    TensorView misaligned{static_cast<char*>(memory) + 1, original.shape(),
                          original.dtype(), cpu, original.strides()};
    reject_component(misaligned, Code::kInvalidMetadata,
                     "Nonempty components require natural alignment.");
    TensorView end{
        reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 7),
        original.shape(), original.dtype(), cpu, original.strides()};
    reject_component(end, Code::kIntegerOverflow,
                     "Component address endpoints cannot wrap.");
    for (std::ptrdiff_t stride : {-1, 0, 2}) {
      std::array<std::ptrdiff_t, 2> strides{3, 1};
      strides[original.ndim() - 1] = stride;
      TensorView changed{
          memory, original.shape(), original.dtype(), cpu,
          TensorView::StridesView{strides.data(), original.ndim()}};
      reject_component(
          changed, Code::kUnsupportedCapability,
          "Negative or noncontiguous component strides are rejected.",
          original.ndim() - 1);
    }
    std::array<std::ptrdiff_t, 3> extra_strides{3, 1, 1};
    for (auto stride_rank : {std::size_t{0}, original.ndim() + 1}) {
      TensorView mismatch{
          memory, original.shape(), original.dtype(), cpu,
          TensorView::StridesView{extra_strides.data(), stride_rank}};
      reject_component(mismatch, Code::kInvalidMetadata,
                       "Stride rank must match component rank.");
    }
    if (role != 3) {
      TensorView wrong_rank{memory,
                            role == 0
                                ? TensorView::Shape{6}
                                : TensorView::Shape{1, original.shape()[0]},
                            original.dtype(), cpu};
      reject_component(wrong_rank, Code::kInvalidMetadata,
                       "Index component rank is checked.");
      auto shape = role == 0 ? TensorView::Shape{2, 2}
                             : TensorView::Shape{original.shape()[0] - 1};
      TensorView wrong_length{memory, shape, original.dtype(), cpu};
      reject_component(wrong_length, Code::kInvalidMetadata,
                       "Index component geometry agrees with shape and nnz.");
      for (const auto device :
           {Device{Device::Type::kCpu, 1}, Device{Device::Type::kNvidia}}) {
        TensorView mismatch{memory, original.shape(), original.dtype(), device,
                            original.strides()};
        reject_component(mismatch, Code::kDeviceMismatch,
                         "Both component device type and index must agree.");
      }
    }
  }
}

void TestDeviceDescriptors(void* memory) {
  Optional out;
  for (int type = 0; type < static_cast<int>(Device::Type::kCount); ++type) {
    const Device device{static_cast<Device::Type>(type), 3};
    TensorView indices{memory, {2, 1}, DataType::kInt64, device, {1, 1}};
    TensorView values{memory, {1}, DataType::kFloat32, device, {1}};
    Expect(SparseTensorView::CreateCoo({4, 5}, indices, values, out),
           Code::kSuccess, out,
           "Device descriptions require no enabled backend.");
    if (!out) return;
    if (out) Check(out->device() == device, "Device index is preserved.");
  }
  for (const auto device :
       {Device{static_cast<Device::Type>(-1)}, Device{Device::Type::kCount},
        Device{Device::Type::kCpu, -1}}) {
    TensorView indices{memory, {2, 1}, DataType::kInt64, device, {1, 1}};
    TensorView values{memory, {1}, DataType::kFloat32, device, {1}};
    ExpectFailure(SparseTensorView::CreateCoo({4, 5}, indices, values, out),
                  Code::kInvalidMetadata, Component::kIndices, out,
                  "Invalid matching device descriptors are rejected.");
  }
}

// Deleted rvalue accessors must remain unavailable through overload resolution.
#define CHECK_BORROWED_ACCESSOR(name)                                         \
  template <typename T, typename = void>                                      \
  struct CanCall_##name : std::false_type {};                                 \
  template <typename T>                                                       \
  struct CanCall_##                                                           \
      name<T, std::void_t<decltype(std::declval<T>().name(                    \
                  std::declval<const TensorView*&>()))>> : std::true_type {}; \
  static_assert(CanCall_##name<const SparseTensorView&>::value);              \
  static_assert(!CanCall_##name<SparseTensorView&&>::value);                  \
  static_assert(!CanCall_##name<const SparseTensorView&&>::value)

CHECK_BORROWED_ACCESSOR(indices);
CHECK_BORROWED_ACCESSOR(raw_indices);
CHECK_BORROWED_ACCESSOR(crow_indices);
CHECK_BORROWED_ACCESSOR(col_indices);
CHECK_BORROWED_ACCESSOR(values);
CHECK_BORROWED_ACCESSOR(raw_values);
#undef CHECK_BORROWED_ACCESSOR

void TestApiBoundary() {
  static_assert(!tensor_view_detail::IsTensorLike<SparseTensorView>::value,
                "Sparse descriptors must not satisfy dense TensorLike.");
  static_assert(std::is_nothrow_copy_constructible_v<SparseTensorView>);
  static_assert(std::is_nothrow_move_constructible_v<SparseTensorView>);
  static_assert(!std::is_copy_assignable_v<SparseTensorView>);
  static_assert(!std::is_move_assignable_v<SparseTensorView>);
}

}  // namespace

int main() {
  // On Linux, any component data access faults immediately. Other platforms
  // still run every metadata case against suitably aligned local storage.
#ifdef __linux__
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    std::perror("sysconf");
    return 2;
  }
  void* memory = mmap(nullptr, static_cast<std::size_t>(page_size), PROT_NONE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (memory == MAP_FAILED) {
    std::perror("mmap");
    return 2;
  }
#else
  alignas(std::max_align_t) unsigned char storage[128]{};
  void* memory = storage;
#endif
  TestBasics(memory);
  TestEmptyAndSingleton(memory);
  TestOverflow(memory);
  TestMetadataOwnershipAndAllocations(memory);
  TestDetailedFailures(memory);
  TestDeviceDescriptors(memory);
  TestApiBoundary();
#ifdef __linux__
  Check(munmap(memory, static_cast<std::size_t>(page_size)) == 0,
        "Protected component storage is released.");
#endif
  std::printf("Sparse metadata checks: %d, failures: %d\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
