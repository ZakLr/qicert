// selftest_main.cpp — qicert C++ skeleton self-test
// Prints the skeleton banner and exercises the one tested property
// (identity cores reproduce the input). Makes no performance statement.

#include "qicert/tt_matvec.hpp"

#include <cstdio>

int main() {
    std::printf("qicert-cpp SKELETON (Phase-2 lever; no performance claims)\n");
    const int rc = qicert::tt_matvec_identity_selftest();
    if (rc == 0) {
        std::printf("selftest: identity-core property OK\n");
        return 0;
    }
    std::printf("selftest: FAILED (rc=%d)\n", rc);
    return 1;
}
