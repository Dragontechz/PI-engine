#include <stdio.h>
#include <math.h>
#include "verlet3d.h"
int main(void) {
    static World w;
    memset(&w, 0, sizeof(w)); rng_seed(&w, 1);
    int id = add_body(&w, 0, 50, 0, 0.3f, 1.0f, 0.0f, 0);
    const float dt = FIXED_DT;
    const float gstep = GRAVITY * dt * dt;   /* per-step velocity gain in verlet */
    for (int s = 1; s <= 100; s++) world_step(&w);
    float vA = w.bodies[id].pos.y - w.bodies[id].prev.y;
    /* analytic with damping: falling velocity is negative. */
    float d = 0.998f;
    float vE = -gstep * (1.0f - (float)pow(d, 100)) / (1.0f - d);
    printf("  velocity after 100 steps: actual=%.5f  analytic(with damping)=%.5f  err=%.3f%%\n",
           vA, vE, fabsf(vA - vE) / vE * 100);
    int ok = fabsf(vA - vE) / vE < 0.02f;
    printf("[%s] T3: damped Verlet gravity velocity\n", ok ? "PASS" : "FAIL");
    return !ok;
}
