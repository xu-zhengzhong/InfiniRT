# Compatibility

This page defines the intended documentation and compatibility boundary for
InfiniRT consumers.

## Stable User Entry

Use:

```cpp
#include <infini/rt.h>
```

The stable user-facing surface is:

- `infini::rt::Device`
- `infini::rt::DataType`
- `infini::rt::TensorView`
- `infini::rt::set_runtime_device_type`
- `infini::rt::runtime_device_type`
- `infini::rt::runtime` types, dispatching functions, and constants documented in
  [Runtime API](api/runtime.md)

The experimental sparse descriptor surface (`TensorLayout`, `SparseTensorView`,
`ValidationResult`, `ValidationCode`, and `ValidationComponent`) is also
available through this entry. Its supported scope and validation boundary are
specified under [Sparse Contract and Release Boundary](#sparse-contract-and-release-boundary).

## Implementation-Facing Headers

Installed headers under `infini/rt/detail/*` are generated to support the public
entry header. They are not intended as direct user includes.

Backend wrapper headers such as `infini/rt/cpu/runtime_.h` expose
backend-specific `runtime::Runtime<Device::Type::...>` specializations. They are
useful for backend tests and advanced integrations, but ordinary consumers
should prefer `<infini/rt.h>`.

## Generated Headers

Generated public headers depend on the backend options used during CMake
configuration. A consumer should compile against the installed prefix produced
by the same configured build.

## ABI Notes

InfiniRT currently exposes a C++ API. Consumers should treat the installed
headers and `libinfinirt.so` as a matching pair from the same build or release.

`TensorView::Shape` and `TensorView::Strides` are concrete vector-like C++
aliases using inline capacity 8. `TensorView` stores ranks 0 through 8 inline
and uses owned heap fallback at rank 9 and above. This representation changes
`TensorView` layout and is an API/ABI compatibility break from the previous
`std::vector` aliases. Consumers must rebuild after this alias or layout change
and must not mix headers and libraries from different builds.

Existing construction from `std::vector` remains supported, but code that
requires the exact `std::vector` alias must adapt. On lvalues, `shape()` and
`strides()` now return typed borrowed contiguous views by value; callers that
need ownership should explicitly materialize `TensorView::Shape` or
`TensorView::Strides`. The owning aliases support the common
`Strides(count, value)` construction used by downstream metadata code.

## Sparse Contract and Release Boundary

**Status: RT1 implementation, RT2 installed export, and RT3 metadata tests implemented.**
The [SparseTensorView contract](api/core-types.md#sparsetensorview-d0-contract)
is implemented in `src/tensor_layout.h` and `src/sparse_tensor_view.h` / `.cc`.
RT2 exports this additive, experimental surface through `<infini/rt.h>` and
verifies COO/CSR construction and component access against the installed
library. No sparse operators, C ABI, owning tensor framework, autograd engine,
or Torch/sparse vendor-library dependency is introduced by RT1–RT3.

`TensorView` retains its dense meaning, object layout, hash, and equality.
`SparseTensorView` is independent and must not satisfy dense interfaces by
inventing a single aggregate data pointer or strides. The initial sparse
representation accepts only two-dimensional, contiguous-component COO/CSR,
with int64 COO indices or matching int32/int64 CSR indices. CSC, BSR, BSC,
batch, and hybrid representations remain explicitly unsupported. Describing
components on a device does not imply that Ops can validate or compute on
that device. Provider support must be declared and tested separately.

Sparse factories validate metadata without reading data or synchronizing.
The sparse `ValidationResult` contract is separate from existing runtime error
types, and external invalid input is rejected in Release as well as Debug.
Index-content validation and checked operator entry belong to InfiniOps;
trusted input requires a validated, unchanged structure. Neither coalesced
metadata nor descriptor equality proves index-content identity. Sparse
metadata caches must not be treated as reusable structure-analysis plans.

Output buffers remain caller-owned. No operator or bridge may silently
replace a buffer, use capacity as valid `nnz`, or publish a pending/failed
dynamic result. All component owners and execution resources survive through
completion on every consuming stream. These obligations also apply to
framework wrappers and to interoperation with independently allocated
components; copying a view does not transfer ownership.

RT2 connects the new types to header generation, include rewriting, umbrella
export, and installation. The install smoke uses only `<infini/rt.h>` and the
installed prefix for RT dependencies. Header and generator changes trigger
regeneration on the next CMake build. RT3 adds
`tests/test_sparse_tensor_view.cc` for metadata, overflow, device, empty-component,
and lifetime checks, and extends the installed consumer with error and boundary
cases. These checks remain active in Release. Existing `test_core`,
`test_tensor_view_allocations`, and dense footprint benchmarks provide the dense
regression coverage. These tests establish no sparse compute capability; Ops
content validation and operators remain unimplemented. D0 remains a
documentation-only contract.

The RT headers and library must remain a matching build/release pair, and
Ops headers, library, and wheel must be built against an identified matching
RT revision or installation package. Cross-repository integration records
that revision explicitly rather than depending on an unspecified latest RT.
Preserving the current dense representation is a regression constraint, not
a new promise of binary compatibility between arbitrary releases. Sparse
releases must state their supported layouts, devices, dtypes, index widths,
and experimental API status; unsupported combinations must fail explicitly
without implicit densification, device transfer, or index narrowing.
