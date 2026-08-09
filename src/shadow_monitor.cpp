#include "qicert/shadow_monitor.hpp"

#include <stdexcept>

namespace qicert {

Alarm monitor_step(const ShadowMonitorConfig&, const double*, std::size_t,
                   int*) {
  // Phase-0 stub: median-of-means + PDU hygiene land with N10.
  throw std::logic_error("monitor_step: not implemented until N10.");
}

}  // namespace qicert
