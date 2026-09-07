#include <stdio.h>
#include "verlet3d.h"
int main(void) {
    static World w;
    world_reset(&w, 42);
    printf("bodies=%d cons=%d boxes=%d\n", w.bodyCount, w.conCount, w.boxCount);
    for (int s = 0; s <= 300; s++) {
        if (s % 60 == 0)
            printf("step %3d: ball0 y=%7.3f  head0 y=%7.3f  cube0corner y=%7.3f\n",
                   s, w.bodies[60].pos.y, w.bodies[37].pos.y, w.bodies[21].pos.y);
        world_step(&w);
    }
    return 0;
}
