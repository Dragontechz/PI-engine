#include "gbuffer.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static V3 camera_forward(const RtGBufferCamera *camera) {
    float cp = cosf(camera->pitch);
    return vnorm(v3(cosf(camera->yaw) * cp,
                    sinf(camera->pitch),
                    sinf(camera->yaw) * cp));
}

static void camera_basis(const RtGBufferCamera *camera, V3 *forward,
                         V3 *right, V3 *up) {
    *forward = camera_forward(camera);
    *right = vnorm(vcross(*forward, v3(0.0f, 1.0f, 0.0f)));
    *up = vcross(*right, *forward);
}

static int project_pixel(const RtGBufferCamera *camera, V3 point,
                         int width, int height, float *x, float *y) {
    V3 forward, right, up;
    camera_basis(camera, &forward, &right, &up);
    V3 relative = vsub(point, camera->position);
    float depth = vdot(relative, forward);
    if (!(depth > RT_EPSILON) || !isfinite(depth)) return 0;
    float aspect = (float)width / (float)height;
    float fov = camera->fov > 1.0f ? camera->fov : 72.0f;
    float tan_h = tanf(fov * 3.14159265358979323846f / 360.0f);
    float ndc_x = vdot(relative, right) / (depth * aspect * tan_h);
    float ndc_y = vdot(relative, up) / (depth * tan_h);
    *x = (ndc_x * 0.5f + 0.5f) * (float)width - 0.5f;
    *y = (1.0f - (ndc_y * 0.5f + 0.5f)) * (float)height - 0.5f;
    return isfinite(*x) && isfinite(*y);
}

int rt_gbuffer_create(RtGBuffer *buffer, int width, int height) {
    if (!buffer || width <= 0 || height <= 0) return 0;
    memset(buffer, 0, sizeof *buffer);
    size_t count = (size_t)width * height;
    buffer->depth = (float *)malloc(count * sizeof *buffer->depth);
    buffer->normal = (V3 *)malloc(count * sizeof *buffer->normal);
    buffer->object = (int *)malloc(count * sizeof *buffer->object);
    buffer->motion_x = (float *)malloc(count * sizeof *buffer->motion_x);
    buffer->motion_y = (float *)malloc(count * sizeof *buffer->motion_y);
    buffer->valid_motion = (unsigned char *)malloc(count * sizeof *buffer->valid_motion);
    if (!buffer->depth || !buffer->normal || !buffer->object ||
        !buffer->motion_x || !buffer->motion_y || !buffer->valid_motion) {
        rt_gbuffer_destroy(buffer);
        return 0;
    }
    buffer->width = width;
    buffer->height = height;
    return 1;
}

void rt_gbuffer_destroy(RtGBuffer *buffer) {
    if (!buffer) return;
    free(buffer->depth);
    free(buffer->normal);
    free(buffer->object);
    free(buffer->motion_x);
    free(buffer->motion_y);
    free(buffer->valid_motion);
    memset(buffer, 0, sizeof *buffer);
}

int rt_gbuffer_render(const Scene *scene, const RtGBufferCamera *current,
                      const RtGBufferCamera *previous, RtGBuffer *buffer) {
    if (!scene || !current || !previous || !buffer || buffer->width <= 0 ||
        buffer->height <= 0 || !buffer->depth || !buffer->normal ||
        !buffer->object || !buffer->motion_x || !buffer->motion_y ||
        !buffer->valid_motion) return 0;
    V3 forward, right, up;
    camera_basis(current, &forward, &right, &up);
    float aspect = (float)buffer->width / (float)buffer->height;
    float fov = current->fov > 1.0f ? current->fov : 72.0f;
    float tan_h = tanf(fov * 3.14159265358979323846f / 360.0f);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < buffer->height; y++) {
        for (int x = 0; x < buffer->width; x++) {
            int index = y * buffer->width + x;
            float u = ((float)x + 0.5f) / (float)buffer->width * 2.0f - 1.0f;
            float v = 1.0f - ((float)y + 0.5f) / (float)buffer->height * 2.0f;
            V3 direction = vnorm(vadd(forward, vadd(
                vscale(right, u * aspect * tan_h), vscale(up, v * tan_h))));
            Hit hit;
            if (!rt_trace(scene, current->position, direction, 1e30f, &hit)) {
                buffer->depth[index] = FLT_MAX;
                buffer->normal[index] = v3(0, 0, 0);
                buffer->object[index] = -1;
                buffer->motion_x[index] = 0.0f;
                buffer->motion_y[index] = 0.0f;
                buffer->valid_motion[index] = 0;
                continue;
            }
            V3 point = vadd(current->position, vscale(direction, hit.t));
            buffer->depth[index] = vdot(vsub(point, current->position), forward);
            buffer->normal[index] = hit.normal;
            buffer->object[index] = hit.object;
            float current_x, current_y, previous_x, previous_y;
            int current_ok = project_pixel(current, point, buffer->width,
                                           buffer->height, &current_x, &current_y);
            int previous_ok = project_pixel(previous, point, buffer->width,
                                            buffer->height, &previous_x, &previous_y);
            if (!current_ok || !previous_ok) {
                buffer->motion_x[index] = 0.0f;
                buffer->motion_y[index] = 0.0f;
                buffer->valid_motion[index] = 0;
            } else {
                /* Pixel coordinates use y-down texture space. This makes a
                 * rightward camera yaw produce leftward background motion. */
                buffer->motion_x[index] = current_x - previous_x;
                buffer->motion_y[index] = current_y - previous_y;
                buffer->valid_motion[index] = 1;
            }
        }
    }
    return 1;
}

void rt_gbuffer_dilate_motion(RtGBuffer *buffer, int radius) {
    if (!buffer || radius < 1 || radius > 2 || !buffer->motion_x ||
        !buffer->motion_y || !buffer->valid_motion) return;
    size_t count = (size_t)buffer->width * buffer->height;
    float *x = (float *)malloc(count * sizeof *x);
    float *y = (float *)malloc(count * sizeof *y);
    unsigned char *valid = (unsigned char *)malloc(count * sizeof *valid);
    if (!x || !y || !valid) {
        free(x); free(y); free(valid);
        return;
    }
    memcpy(x, buffer->motion_x, count * sizeof *x);
    memcpy(y, buffer->motion_y, count * sizeof *y);
    memcpy(valid, buffer->valid_motion, count * sizeof *valid);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int py = 0; py < buffer->height; py++) {
        for (int px = 0; px < buffer->width; px++) {
            int best = py * buffer->width + px;
            float best_len = valid[best]
                ? x[best] * x[best] + y[best] * y[best] : -1.0f;
            float best_depth = buffer->depth[best];
            for (int oy = -radius; oy <= radius; oy++) {
                int sy = py + oy;
                if (sy < 0 || sy >= buffer->height) continue;
                for (int ox = -radius; ox <= radius; ox++) {
                    int sx = px + ox;
                    if (sx < 0 || sx >= buffer->width) continue;
                    int candidate = sy * buffer->width + sx;
                    if (!valid[candidate]) continue;
                    float length = x[candidate] * x[candidate] +
                                   y[candidate] * y[candidate];
                    float depth = buffer->depth[candidate];
                    if (length > best_len ||
                        (length == best_len && depth < best_depth)) {
                        best = candidate;
                        best_len = length;
                        best_depth = depth;
                    }
                }
            }
            int output = py * buffer->width + px;
            buffer->motion_x[output] = best_len >= 0.0f ? x[best] : 0.0f;
            buffer->motion_y[output] = best_len >= 0.0f ? y[best] : 0.0f;
            buffer->valid_motion[output] = (unsigned char)(best_len >= 0.0f);
        }
    }
    free(x); free(y); free(valid);
}

int rt_gbuffer_reject_history(float current_depth, float history_depth,
                              V3 current_normal, V3 history_normal,
                              float depth_tau) {
    if (!(current_depth > 0.0f) || !(history_depth > 0.0f) ||
        !isfinite(current_depth) || !isfinite(history_depth)) return 1;
    if (depth_tau <= 0.0f) depth_tau = 0.02f;
    if (fabsf(current_depth - history_depth) > depth_tau * current_depth) return 1;
    float current_len = vdot(current_normal, current_normal);
    float history_len = vdot(history_normal, history_normal);
    if (current_len < 0.5f || history_len < 0.5f) return 0;
    return vdot(vnorm(current_normal), vnorm(history_normal)) < 0.8f;
}

float rt_gbuffer_mean_motion(const RtGBuffer *buffer) {
    if (!buffer || !buffer->motion_x || !buffer->motion_y) return 0.0f;
    double sum = 0.0;
    size_t count = (size_t)buffer->width * buffer->height;
    for (size_t i = 0; i < count; i++)
        if (buffer->valid_motion[i])
            sum += sqrt((double)buffer->motion_x[i] * buffer->motion_x[i] +
                        (double)buffer->motion_y[i] * buffer->motion_y[i]);
    return (float)(sum / (double)(count > 0 ? count : 1));
}

float rt_gbuffer_valid_motion_fraction(const RtGBuffer *buffer) {
    if (!buffer || !buffer->valid_motion) return 0.0f;
    size_t count = (size_t)buffer->width * buffer->height;
    size_t valid = 0;
    for (size_t i = 0; i < count; i++) valid += buffer->valid_motion[i] != 0;
    return count > 0 ? (float)valid / (float)count : 0.0f;
}
