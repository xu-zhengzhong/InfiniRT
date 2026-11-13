# Core Types

Core types are available through:

```cpp
#include <infini/rt.h>
```

## Device

`infini::rt::Device` identifies a backend type and an index.

```cpp
infini::rt::Device device{infini::rt::Device::Type::kCpu, 0};
auto type = device.type();
auto index = device.index();
```

Known device types include:

- `Device::Type::kCpu`
- `Device::Type::kNvidia`
- `Device::Type::kIluvatar`
- `Device::Type::kMetax`
- `Device::Type::kMars`
- `Device::Type::kMoore`
- `Device::Type::kHygon`
- `Device::Type::kCambricon`
- `Device::Type::kAscend`

## DataType

`infini::rt::DataType` describes tensor element types.

Supported values include signed integers, unsigned integers, reduced precision
floating point, and standard floating point types:

- `kInt8`, `kInt16`, `kInt32`, `kInt64`
- `kUInt8`, `kUInt16`, `kUInt32`, `kUInt64`
- `kFloat16`, `kBFloat16`, `kFloat32`, `kFloat64`

## TensorView

`infini::rt::TensorView` is a non-owning description of tensor memory that owns
its shape and stride metadata.

```cpp
std::vector<float> data(16);

infini::rt::TensorView tensor{
    data.data(),
    std::vector<std::size_t>{4, 4},
    infini::rt::DataType::kFloat32,
    infini::rt::Device{infini::rt::Device::Type::kCpu, 0},
    std::vector<std::ptrdiff_t>{4, 1}};

auto elements = tensor.numel();
auto bytes_per_element = tensor.element_size();
auto contiguous = tensor.IsContiguous();
```

`TensorView` stores:

- data pointer
- shape
- data type
- device
- strides

It does not own the tensor data it references. Shape and strides are stored
inline for ranks 0 through 8; rank 9 and above use owned heap fallback storage.
Construction from `std::vector` and other compatible contiguous ranges remains
supported.

On an lvalue `TensorView`, `shape()` and `strides()` return lightweight
contiguous views by value. Those views borrow metadata from the `TensorView` and
must not outlive it. Calling the accessors on an rvalue returns owning metadata
so a view cannot dangle from a temporary.

## SparseTensorView: D0 Contract

**Status: RT1 source implementation available; installed API pending RT2/RT3.**
`infini::rt::TensorLayout`, `infini::rt::SparseTensorView`, and
`ValidationResult` are implemented in `src/tensor_layout.h` and
`src/sparse_tensor_view.h` / `.cc` under the D0 contract below. They are not yet
exported through `<infini/rt.h>`, and no sparse operators are implemented. See
[Compatibility](../compatibility.md#sparse-contract-and-release-boundary) for the
implementation and release boundary.

`SparseTensorView` is a separate, copyable, non-owning descriptor. It owns its
logical shape metadata and copies the metadata of its dense component views;
it must not borrow metadata from temporary `TensorView` objects. Copying a
sparse descriptor neither copies component data nor extends its lifetime.
Construction performs no device allocation. The existing dense `TensorView`
object layout, hash, equality, and dense operations remain unchanged.

`TensorLayout` distinguishes COO, CSR, CSC, BSR, and BSC. Only COO and CSR
construction is in the first implementation scope. Reserved layout values do
not imply available factories or operators. The initial factories accept an
explicit two-dimensional logical shape `[M, K]`, with `sparse_dim = 2`,
`dense_dim = 0`, and `batch_dim = 0`; batch, hybrid, and block layouts are rejected
until separately implemented. Logical shape is never inferred from indices.

| Property | COO | CSR |
| --- | --- | --- |
| Index components | Contiguous `indices` of shape `[2, N]`, dtype `int64` | Contiguous `crow_indices` of shape `[M + 1]` and `col_indices` of shape `[N]`, both `int32` or both `int64` |
| Values | Contiguous `values` of shape `[N]` | Contiguous `values` of shape `[N]` |
| Stored count `nnz` | `N`, including explicit zeros and duplicate coordinates | `N`, including explicit zeros |
| Normalization metadata | Declared COO coalesced state | No COO coalesced flag; canonical CSR content is checked by Ops |
| Empty representation | `indices` shape `[2, 0]`, `values` shape `[0]` | `crow_indices` still has `M + 1` elements; other components have length zero |

The descriptor exposes logical shape, layout, dimension decomposition, value
dtype, index dtype, the common full device `(type, index)`, and `nnz`. Value
dtype describes storage; it does not promise that an Ops provider can compute
with that dtype. The first Ops CPU scope is float32/float64. All components
must reside on the same full device. In future compressed batch layouts `nnz`
is the common count per batch, and in block layouts it counts blocks.

`numel` is the checked product of the logical dimensions, including zero-sized
dimensions. It is not `nnz`, allocation capacity, or sparse storage size.
Component sizes and byte counts are calculated separately with overflow
checks. A sparse descriptor has no aggregate `data()`, `strides()`,
`IsContiguous()`, pointer-arithmetic slicing, or unconditional `T()` operation.
These dense facilities apply only to its individual components.

COO `indices()` and `values()` require a descriptor declared coalesced. Explicit
low-level raw component access is available to import and validate an
uncoalesced COO without sorting, merging, or allocating. A declared coalesced
state is not evidence that the data was validated. Empty COO construction
preserves the caller declaration; it does not silently promote it to coalesced.
An omitted coalesced attribute defaults to false, including for empty COO.
CSR uses
`crow_indices()`, `col_indices()`, and `values()` without a COO-style coalesce
requirement. Invalid layout or state access must return a checked failure; it
must not depend on a Release-disabled assertion.

### RT1 Source API

Source-tree consumers include `"sparse_tensor_view.h"` and link the matching
InfiniRT library. RT2 will add public-header generation and installation;
RT3 will validate the installed consumer. This example uses caller-owned
storage that remains alive throughout the view's use:

```cpp
#include <cstdint>
#include <optional>

#include "sparse_tensor_view.h"

int main() {
  using namespace infini::rt;
  const Device cpu{Device::Type::kCpu};
  std::int64_t coordinates[] = {0, 1, 1, 2};
  float data[] = {3.0f, 4.0f};
  const TensorView indices{coordinates, {2, 2}, DataType::kInt64, cpu};
  const TensorView values{data, {2}, DataType::kFloat32, cpu};
  std::optional<SparseTensorView> view;

  const auto status = SparseTensorView::CreateCoo({2, 3}, indices, values, view);
  if (!status) return 1;

  const TensorView* raw = nullptr;
  const auto access = view->raw_values(raw);
  return access.ok() && raw->data() == data ? 0 : 1;
}
```

`CreateCoo(shape, indices, values, out, coalesced = false)` and
`CreateCsr(shape, crow_indices, col_indices, values, out)` return
`ValidationResult` and populate `std::optional<SparseTensorView>` only after
successful metadata checks. A failed call clears `out`. The general
`Create(layout, shape, components, out, coalesced = false)` takes component
pointers in COO `{indices, values}` or CSR `{crow_indices, col_indices, values}`
order and rejects unsupported layouts and incorrect component counts. Shape
is a `ShapeView`, `SparseTensorView::Shape` (`std::array<Size, 2>`), or
initializer list of `TensorView::Size` extents.

`ValidationResult` has `ValidationCode code`, `ValidationComponent component`,
and optional `std::size_t position`; `ok()` and explicit boolean conversion
check success. The RT codes are the metadata subset of D0, named with the
repository's `k` prefix, such as `kIntegerOverflow` and `kInvalidState`.
Successful results have no error position. Component accessors return this
same result and write a borrowed `const TensorView*`, clearing it on failure.
Those pointers must not outlive the descriptor; component access on a
temporary descriptor is disabled. `is_coalesced(bool& out)` returns
`kInvalidState` on CSR.

The descriptor can be copy- or move-constructed; assignment is disabled like
`TensorView`. Its `shape()` returns an owning `std::array<Size, 2>` by value.
All supported metadata fits inline, so factory construction and descriptor
copying allocate no host or device memory. `kAllocationFailure` is reserved
for future representations that need metadata allocation. Empty components
accept nonnegative strides without accessing storage; singleton dimensions
do not constrain contiguous strides. Negative strides are always rejected.

### Metadata Validation and Content Validation

The COO and CSR factories return a checkable `ValidationResult` and publish a
view only on success. They do not throw C++ exceptions. Host metadata
allocation failure returns `AllocationFailure` without publishing a partially
constructed view. A successful factory
result means metadata is valid; it does not establish index-content validity
or provider support. RT does not scan component data, submit a device kernel,
or synchronize a stream during construction.

RT checks the supported layout and dimension decomposition, component count,
rank and shape, index dtype agreement, complete device agreement, contiguous
component layout, non-null and naturally aligned pointers for nonempty typed
components, arithmetic representability, and index-width limits. Alignment is
checked before any content read; misaligned components return `InvalidMetadata`. As an explicit conservative initial
limit, each logical extent and `N` must be no larger than the signed maximum
of the chosen index dtype (`INT32_MAX` or `INT64_MAX`). Negative strides and
unsupported noncontiguous components are rejected. `M + 1`, component element/byte counts,
logical element counts, and address-range calculations must not wrap. Invalid
metadata, unsupported representations, and integer overflow remain distinct
failure categories. Empty components may have null pointers and must not be
dereferenced. In particular, CSR `[0, K]` still requires a one-element row
pointer component. A zero logical dimension cannot contain stored entries.

InfiniOps owns full content validation: COO coordinate bounds and the truth
of a coalesced declaration; CSR pointer endpoints and monotonicity, column
bounds, and strict column ordering within each row. A valid empty CSR has all
row pointers zero, but RT cannot establish that by metadata alone. A false
coalesced claim, invalid row pointer, or out-of-range coordinate must be
rejected by the Ops checked entry before an unsafe computation can run.
Device content is checked on the device or through an explicitly synchronous
path, never by host dereference of a device pointer.

`ValidationResult` carries a status code, a component identifier, and an
optional error position; it does not borrow an error string. RT returns the
metadata-validation categories. Ops preserves those codes and adds content
and execution failures at its boundary. Numeric enum values and a binary
ABI are not frozen by D0. This is a new sparse contract, not an assertion that
existing `runtime::Error` already defines these errors. External sparse failures
use explicit status propagation in Debug and Release; assertions remain for
internal invariants. Content validity, operator capability, output capacity,
and execution completion are separate conditions. The initial checked import
in Ops performs full content validation by default. Skipping a repeated scan
requires a documented trusted source whose validated structure has remained
unchanged; a raw pointer or `coalesced = true` alone is insufficient.

### Borrowing, Mutation, and Execution Completion

The caller owns every index buffer, values buffer, output buffer, workspace,
and result slot. They must all remain alive until their last consumer has
completed. A view neither releases memory nor serves as an owning group.
Python and other framework bridges must retain all component owners for this
same lifetime, including after the host call returns for asynchronous work.

A producer stream records an event after writing components; each consumer
stream waits for the relevant producer event. All consumers must complete
before structure mutation, reuse, or release. Enqueuing `FreeAsync` on one
stream does not prove that other streams are finished. A backend without the
required event/async support uses an explicitly synchronous restricted path.
Resource release must use the allocating backend and device, independent of
the thread-local backend selected by the releasing thread.

Structure includes indices, shape, layout, `nnz`, block shape, and the COO
coalesced declaration. After changing structure, the caller must discard the
old validation proof, revalidate, and rebuild any dependent structure plan.
The descriptor cannot observe external writes and does not promise automatic
version tracking. Values-only changes also require stream ordering, but need
not invalidate a purely symbolic plan. Inputs remain unchanged between an
Ops Prepare and Execute; count-first dynamic output additionally depends on
unchanged values when those values determine the stored count.

Dynamic output capacity is not a valid sparse `nnz`. Ops owns the
Prepare/Execute/Finalize protocol: the caller provides capacity buffers and a
result slot; a result view is published only after successful completion with
the actual `nnz`. For COO, the final indices must be packed as contiguous
`[2, actual_nnz]`; changing a capacity-shaped view's metadata alone is
insufficient. Output components, workspace, and result-slot write ranges must be mutually
disjoint and must not overlap any input storage in the initial operator scope.
Fixed-pattern CSR output is checked as writable output metadata; its
uninitialized indices are not scanned as if it were an input. InfiniRT
provides the general memory, stream, and event mechanisms, while Ops owns
these execution, alias, and capacity checks.
