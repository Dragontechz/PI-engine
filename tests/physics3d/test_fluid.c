#include <stdio.h>
#include <math.h>
#include "verlet3d.h"
int main(void) {
    static World w;
    world_reset(&w, 42); fflush(stdout);
    /* count fluid particles */
    int fluid = 0;
    for (int i = 0; i < w.bodyCount; i++) if (w.bodies[i].tagged == 3) fluid++;
    printf("fluid particles: %d / %d total bodies\n", fluid, w.bodyCount);
    /* run and measure: water should fall and spread, staying in bounds */
    for (int s = 1; s <= 300; s++) {
        world_step(&w);
        if (s % 100 == 0) {
            float miny=1e9f, maxy=-1e9f, minx=1e9f, maxx=-1e9f, avgx=0;
            int n=0;
            for (int i = 0; i < w.bodyCount; i++) {
                Body *b = &w.bodies[i];
                if (b->tagged != 3) continue;
                if (b->pos.y < miny) miny = b->pos.y;
                if (b->pos.y > maxy) maxy = b->pos.y;
                if (b->pos.x < minx) minx = b->pos.x;
                if (b->pos.x > maxx) maxx = b->pos.x;
                avgx += b->pos.x; n++;
            }
            printf("step %3d: x[%.1f..%.1f] y[%.1f..%.1f] center=%.2f (spreading=%s)\n",
                   s, minx, maxx, miny, maxy, avgx/n, (maxx-minx) > 16.0f ? "YES" : "no");
        }
    }
    /* determinism with fluid */
    static World w2;
    world_reset(&w2, 42);
    for (int s = 0; s < 300; s++) world_step(&w2);
    int same = 1;
    for (int i = 0; i < w.bodyCount; i++) {
        Vector3 d = Vector3Subtract(w.bodies[i].pos, w2.bodies[i].pos);
        if (fabsf(d.x)+fabsf(d.y)+fabsf(d.z) > 1e-4f) { same = 0; break; }
    }
    printf("[%s] fluid determinism: same seed => identical 300-step trajectory\n", same ? "PASS" : "FAIL");
    fflush(stdout); return 0;
}
