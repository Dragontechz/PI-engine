#include "raylib.h"
#include <stdio.h>
int main(void) {
    Image img = LoadImage("render.bmp");
    if (img.data == NULL) { printf("LOAD FAIL\n"); return 1; }
    ExportImage(img, "render.png");
    UnloadImage(img);
    printf("CONVERT OK\n");
    return 0;
}
