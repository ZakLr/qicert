#include "qicert/iqae.hpp"

#include <stdexcept>

namespace qicert {

IQAEInterval iqae_estimate(OracleFn, void*, const IQAEConfig&) {
  // Phase-0 stub: Bayesian IQAE lands with N6 (safety suite).
  throw std::logic_error("iqae_estimate: not implemented until N6.");
}

}  // namespace qicert
