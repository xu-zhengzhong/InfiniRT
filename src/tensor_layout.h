#ifndef INFINI_RT_TENSOR_LAYOUT_H_
#define INFINI_RT_TENSOR_LAYOUT_H_

namespace infini::rt {

enum class TensorLayout {
  kCoo,
  kCsr,
  kCsc,
  kBsr,
  kBsc,
};

}  // namespace infini::rt

#endif
