#include <stdio.h>
#include <math.h>
#include "verlet3d.h"
int main(void) {
    static World w;
    memset(&w, 0, sizeof(w));
    rng_seed(&w, 1);
    int id = add_body(&w, 0, 50, 0, 0.3f, 1.0f, 0.0f, 0);
    const float dt = FIXED_DT;
    float y0 = 50.0f;
    for (int s = 1; s <= 100; s++) world_step(&w);
    float vActual = w.bodies[id].pos.y - w.bodies[id].prev.y;
    float damping = 0.998f;
    float vExpect = -GRAVITY * dt * dt *
                    (1.0f - powf(damping, 100.0f)) / (1.0f - damping);
    float yExpect = y0 - 0.5f * GRAVITY * (100*dt) * (100*dt);
    printf("free-fall after 100 steps:\n");
    printf("  velocity: actual=%.4f/step  expected=%.4f/step  (err %.3f%%)\n",
           vActual, vExpect, fabsf(vActual-vExpect)/vExpect*100);
    printf("  position: actual=%.3f  expected=%.3f\n", w.bodies[id].pos.y, yExpect);
    /* in-flight (step 40) */
    static World w2;
    memset(&w2, 0, sizeof(w2)); rng_seed(&w2, 1);
    int id2 = add_body(&w2, 0, 50, 0, 0.3f, 1.0f, 0.0f, 0);
    for (int s = 1; s <= 40; s++) world_step(&w2);
    float vA = w2.bodies[id2].pos.y - w2.bodies[id2].prev.y;
    float vE = -GRAVITY * dt * dt *
               (1.0f - powf(damping, 40.0f)) / (1.0f - damping);
    printf("  mid-flight(40 steps) velocity: actual=%.4f expected=%.4f (err %.3f%%)\n",
           vA, vE, fabsf(vA-vE)/vE*100);
    int ok = fabsf(vA - vE) / vE < 0.02f;
    printf("[%s] T3 gravity: damped Verlet analytic velocity within 2%%\n", ok ? "PASS" : "FAIL");
    return !ok;
}
