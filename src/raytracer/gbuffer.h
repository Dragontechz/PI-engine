#ifndef RT_GBUFFER_H
#define RT_GBUFFER_H

#include "tracer.h"

/* CPU-side primary-hit data for temporal/reconstruction experiments. This is
 * deliberately independent of PiTSR and is opt-in: the normal renderer does
 * not pay for a second primary-ray pass. Motion is stored in texture pixels
 * (x right, y down), not UV or NDC units. */
typedef struct {
    V3 position;
    float yaw;
    float pitch;
    float fov;
} RtGBufferCamera;

typedef struct {
    int width;
    int height;
    float *depth;       /* forward camera-space depth; FLT_MAX means sky */
    V3 *normal;         /* primary-hit normal, zero for sky */
    int *object;        /* Scene object ID, -1 for sky */
    float *motion_x;    /* current pixel minus previous pixel */
    float *motion_y;
    unsigned char *valid_motion;
} RtGBuffer;

int rt_gbuffer_create(RtGBuffer *buffer, int width, int height);
void rt_gbuffer_destroy(RtGBuffer *buffer);

/* Render primary hits and project each hit into both camera states. The
 * previous camera must be the camera used for the preceding frame. */
int rt_gbuffer_render(const Scene *scene, const RtGBufferCamera *current,
                     const RtGBufferCamera *previous, RtGBuffer *buffer);

/* FSR-style 3x3 dilation. The longest valid vector is selected, preferring
 * the closest depth on ties so foreground silhouettes protect disocclusions. */
void rt_gbuffer_dilate_motion(RtGBuffer *buffer, int radius);

/* Reject a reprojected history sample using relative depth and normal tests. */
int rt_gbuffer_reject_history(float current_depth, float history_depth,
                              V3 current_normal, V3 history_normal,
                              float depth_tau);

float rt_gbuffer_mean_motion(const RtGBuffer *buffer);
float rt_gbuffer_valid_motion_fraction(const RtGBuffer *buffer);

#endif
