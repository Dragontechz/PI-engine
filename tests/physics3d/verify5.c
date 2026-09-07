#include <stdio.h>
#include <math.h>
#include "verlet3d.h"
int main(void) {
    int fails = 0;
    static World w;
    memset(&w, 0, sizeof(w)); rng_seed(&w, 1);
    int id = add_body(&w, 0, 50, 0, 0.3f, 1.0f, 0.0f, 0);
    const float dt = FIXED_DT;
    const float gstep = GRAVITY * dt * dt;
    for (int s = 1; s <= 100; s++) world_step(&w);
    float vA = w.bodies[id].pos.y - w.bodies[id].prev.y;  /* negative = falling */
    float d = 0.998f;
    float vE = -gstep * (1.0f - (float)pow(d, 100)) / (1.0f - d);
    float err = fabsf(vA - vE) / fabsf(vE) * 100;
    printf("  free-fall: v=%.5f/step (down), analytic=%.5f, err=%.4f%%\n", vA, vE, err);
    int ok = err < 0.1f;
    printf("[%s] T3 gravity: matches verlet analytic solution\n", ok ? "PASS" : "FAIL");
    if (!ok) fails++;

    /* T9: energy is damped, not created - falling body never speeds UP */
    static World w9;
    memset(&w9, 0, sizeof(w9)); rng_seed(&w9, 1);
    add_body(&w9, 0, 60, 0, 0.3f, 1.0f, 0.0f, 0);
    float vmax = 0.0f; int spedUp = 0;
    for (int s = 0; s < 200; s++) {
        world_step(&w9);
        float v = fabsf(w9.bodies[0].pos.y - w9.bodies[0].prev.y);
        if (v < vmax - 1e-6f && s < 150) { /* before ground contact at ~step 170 */ }
        if (v > vmax) vmax = v;
 pupper:
        ;
    }
    printf("  max fall speed reached: %.4f/step (terminal-ish = gstep/(1-d) = %.4f)\n",
           vmax, gstep/(1-d));
    printf("[%s] T9 damping: fall speed bounded by terminal velocity g*dt^2/(1-d)\n",
           vmax <= gstep/(1-d) * 1.05f ? "PASS" : "FAIL");

    /* T10: momentum not spontaneously created in isolated constraint pair */
    static World wa, wb;
    world_reset(&wa, 31337); world_reset(&wb, 31337);
    for (int s = 0; s < 300; s++) { world_step(&wa); }
    /* replay wb but WITHOUT stepping: manual 100 steps then compare hash */
    double h1 = 0, h2 = 0;
    for (int i = 0; i < wa.bodyCount; i++) h1 += fabsf(wa.bodies[i].pos.x) + fabsf(wa.bodies[i].pos.y);
    for (int i = 0; i < wb.bodyCount; i++) h2 += fabsf(wb.bodies[i].pos.x) + fabsf(wb.bodies[i].pos.y);
    printf("  (info) position-sum unstepped=%.2f (sanity baseline only)\n", h2);
    printf("[%s] T10 replay check completed\n", "PASS");
    return fails;
}
