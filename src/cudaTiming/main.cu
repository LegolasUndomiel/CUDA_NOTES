#include "cudaTimer.cuh"

int main(int argc, char const *argv[]) {
    cudaTimer timer;

    timer.start();

    // 计时

    float elapsedTime = timer.stop();
    printf("Elapsed time: %g ms\n", elapsedTime);

    return 0;
}
