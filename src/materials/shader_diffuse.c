#include "materials.h"

#define RTM_PI 3.14159265358979323846f

RtmVec3 rtm_lambert(RtmVec3 color) {
    return rtm_scale(color, 1.0f / RTM_PI);
}
