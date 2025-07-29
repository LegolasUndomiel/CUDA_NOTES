#include "cudaTimer.h"

int main(int argc, char const *argv[]) {
    cudaTimer timer;

    timer.start();

    // 计时

    timer.stop();

    return 0;
}
