/* Scene builders for the ray tracer (see tracer.h). */
#include "tracer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void need(Scene *s, int n);

static void mesh_triangle_bounds(const RtMesh *mesh, int triangle, V3 *mn, V3 *mx) {
    const int *t = mesh->triangles[triangle];
    V3 a=mesh->vertices[t[0]], b=mesh->vertices[t[1]], c=mesh->vertices[t[2]];
    *mn=v3(fminf(a.x,fminf(b.x,c.x)),fminf(a.y,fminf(b.y,c.y)),fminf(a.z,fminf(b.z,c.z)));
    *mx=v3(fmaxf(a.x,fmaxf(b.x,c.x)),fmaxf(a.y,fmaxf(b.y,c.y)),fmaxf(a.z,fmaxf(b.z,c.z)));
}
static float mesh_centroid(const RtMesh *mesh,int tri,int axis){V3 mn,mx;mesh_triangle_bounds(mesh,tri,&mn,&mx);V3 c=vscale(vadd(mn,mx),.5f);return axis==0?c.x:axis==1?c.y:c.z;}
static int mesh_build_node(RtMesh *mesh,int start,int count){
    int id=mesh->node_count++; RtMeshNode *node=&mesh->nodes[id];
    V3 mn=v3(1e30f,1e30f,1e30f),mx=v3(-1e30f,-1e30f,-1e30f);
    for(int i=start;i<start+count;i++){V3 a,b;mesh_triangle_bounds(mesh,mesh->triangle_indices[i],&a,&b);mn.x=fminf(mn.x,a.x);mn.y=fminf(mn.y,a.y);mn.z=fminf(mn.z,a.z);mx.x=fmaxf(mx.x,b.x);mx.y=fmaxf(mx.y,b.y);mx.z=fmaxf(mx.z,b.z);}
    node->min=mn;node->max=mx;node->left=node->right=-1;node->start=start;node->count=count;
    if(count<=8)return id;
    V3 e=vsub(mx,mn);int axis=e.x>e.y&&e.x>e.z?0:e.y>e.z?1:2;
    for(int i=start+1;i<start+count;i++){int key=mesh->triangle_indices[i];float value=mesh_centroid(mesh,key,axis);int j=i-1;while(j>=start&&mesh_centroid(mesh,mesh->triangle_indices[j],axis)>value){mesh->triangle_indices[j+1]=mesh->triangle_indices[j];j--;}mesh->triangle_indices[j+1]=key;}
    int left=count/2;node->left=mesh_build_node(mesh,start,left);node->right=mesh_build_node(mesh,start+left,count-left);node->count=0;return id;
}

static int mesh_load_ply(RtMesh *mesh, const char *path, V3 position, float scale) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[256];
    int vertices = 0, faces = 0, header = 1;
    while (header && fgets(line, sizeof line, f)) {
        if (sscanf(line, "element vertex %d", &vertices) == 1) continue;
        if (sscanf(line, "element face %d", &faces) == 1) continue;
        if (strncmp(line, "end_header", 10) == 0) header = 0;
    }
    if (vertices <= 0 || vertices > RT_MAX_MESH_VERTICES || faces <= 0 || faces > RT_MAX_MESH_TRIANGLES) { fclose(f); return 0; }
    mesh->vertex_count = vertices;
    mesh->triangle_count = 0;
    for (int i = 0; i < vertices; i++) {
        float x, y, z;
        if (fscanf(f, "%f %f %f", &x, &y, &z) != 3) { fclose(f); return 0; }
        mesh->vertices[i] = vadd(position, vscale(v3(x, y, z), scale));
        fgets(line, sizeof line, f);
    }
    for (int i = 0; i < faces; i++) {
        int n, a, b, c;
        if (fscanf(f, "%d %d %d %d", &n, &a, &b, &c) != 4) break;
        if (n != 3 || a < 0 || b < 0 || c < 0 || a >= vertices || b >= vertices || c >= vertices) continue;
        mesh->triangles[mesh->triangle_count][0] = a;
        mesh->triangles[mesh->triangle_count][1] = b;
        mesh->triangles[mesh->triangle_count][2] = c;
        mesh->triangle_indices[mesh->triangle_count] = mesh->triangle_count;
        mesh->triangle_count++;
    }
    fclose(f);
    mesh->node_count = 0;
    for (int i = 0; i < mesh->triangle_count; i++) mesh->triangle_indices[i] = i;
    if (mesh->triangle_count > 0) mesh_build_node(mesh, 0, mesh->triangle_count);
    return mesh->triangle_count > 0;
}

static int mesh_load_obj(RtMesh *mesh, const char *path, V3 position, float scale) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[1024];
    mesh->vertex_count = 0;
    mesh->triangle_count = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            if (sscanf(line + 2, "%f %f %f", &x, &y, &z) == 3 && mesh->vertex_count < RT_MAX_MESH_VERTICES)
                mesh->vertices[mesh->vertex_count++] = vadd(position, vscale(v3(x, y, z), scale));
        } else if (line[0] == 'f' && line[1] == ' ') {
            int ids[64], count = 0;
            char *cursor = line + 2;
            while (*cursor && count < 64) {
                char *end = cursor;
                while (*end && *end != ' ' && *end != '\t' && *end != '\r' && *end != '\n') end++;
                char saved = *end;
                *end = '\0';
                ids[count++] = atoi(cursor);
                *end = saved;
                cursor = end;
                while (*cursor == ' ' || *cursor == '\t') cursor++;
            }
            for (int i = 1; i + 1 < count && mesh->triangle_count < RT_MAX_MESH_TRIANGLES; i++) {
                int a = ids[0] > 0 ? ids[0] - 1 : mesh->vertex_count + ids[0];
                int b = ids[i] > 0 ? ids[i] - 1 : mesh->vertex_count + ids[i];
                int c = ids[i + 1] > 0 ? ids[i + 1] - 1 : mesh->vertex_count + ids[i + 1];
                if (a < 0 || b < 0 || c < 0 || a >= mesh->vertex_count || b >= mesh->vertex_count || c >= mesh->vertex_count) continue;
                mesh->triangles[mesh->triangle_count][0] = a;
                mesh->triangles[mesh->triangle_count][1] = b;
                mesh->triangles[mesh->triangle_count][2] = c;
                mesh->triangle_count++;
            }
        }
    }
    fclose(f);
    mesh->node_count = 0;
    for (int i = 0; i < mesh->triangle_count; i++) mesh->triangle_indices[i] = i;
    if (mesh->triangle_count > 0) mesh_build_node(mesh, 0, mesh->triangle_count);
    return mesh->triangle_count > 0;
}

static int add_mesh(Scene *s, const char *path, V3 position, float scale, Material material) {
    if (s->mesh_count >= RT_MAX_MESHES || s->count >= RT_MAX_OBJECTS) return -1;
    RtMesh *mesh = &s->meshes[s->mesh_count];
    memset(mesh, 0, sizeof *mesh);
    const char *extension = strrchr(path, '.');
    int loaded = extension && strcmp(extension, ".obj") == 0
        ? mesh_load_obj(mesh, path, position, scale)
        : mesh_load_ply(mesh, path, position, scale);
    if (!loaded) return -1;
    need(s, 1);
    Object *o = &s->objects[s->count++];
    memset(o, 0, sizeof *o);
    o->shape = RT_MESH;
    o->material = material;
    o->geometry.mesh.mesh = s->mesh_count++;
    s->mesh_enabled = 1;
    s->render_meshes = 1;
    o->geometry.mesh.min = v3(1e30f, 1e30f, 1e30f);
    o->geometry.mesh.max = v3(-1e30f, -1e30f, -1e30f);
    for (int i = 0; i < mesh->vertex_count; i++) {
        V3 p = mesh->vertices[i];
        o->geometry.mesh.min.x = fminf(o->geometry.mesh.min.x, p.x); o->geometry.mesh.min.y = fminf(o->geometry.mesh.min.y, p.y); o->geometry.mesh.min.z = fminf(o->geometry.mesh.min.z, p.z);
        o->geometry.mesh.max.x = fmaxf(o->geometry.mesh.max.x, p.x); o->geometry.mesh.max.y = fmaxf(o->geometry.mesh.max.y, p.y); o->geometry.mesh.max.z = fmaxf(o->geometry.mesh.max.z, p.z);
    }
    return s->count - 1;
}

int rt_load_mesh(Scene *s, const char *path, V3 position, float scale, Material material) {
    if (!s || !path) return -1;
    return add_mesh(s, path, position, scale, material);
}

static void need(Scene *s, int n) {
    if (s->count + n > RT_MAX_OBJECTS) {
        fprintf(stderr, "scene overflow: %d + %d > %d\n", s->count, n, RT_MAX_OBJECTS);
        exit(1);
    }
}

static void add_box(Scene *s, V3 mn, V3 mx, Material m) {
    need(s, 1);
    Object *o = &s->objects[s->count++];
    memset(o, 0, sizeof *o);
    o->shape = RT_BOX;
    o->material = m;
    o->geometry.box.min = mn;
    o->geometry.box.max = mx;
}

static int add_sphere(Scene *s, V3 c, float r, Material m) {
    need(s, 1);
    Object *o = &s->objects[s->count++];
    memset(o, 0, sizeof *o);
    o->shape = RT_SPHERE;
    o->material = m;
    o->geometry.sphere.center = c;
    o->geometry.sphere.radius = r;
    return s->count - 1;
}

static void add_cyl(Scene *s, V3 base, float r, float h, Material m) {
    need(s, 1);
    Object *o = &s->objects[s->count++];
    memset(o, 0, sizeof *o);
    o->shape = RT_CYLINDER;
    o->material = m;
    o->geometry.cylinder.base = base;
    o->geometry.cylinder.radius = r;
    o->geometry.cylinder.height = h;
}

static void add_plane(Scene *s, float y, int checker, Material m) {
    need(s, 1);
    Object *o = &s->objects[s->count++];
    memset(o, 0, sizeof *o);
    o->shape = RT_PLANE;
    o->material = m;
    o->geometry.plane.y = y;
    o->geometry.plane.checker = checker;
}

static Material mat(float r, float g, float b, float refl, float shin) {
    Material m = { { r, g, b }, refl, shin, { 0, 0, 0 }, RT_TEX_NONE, 1.0f, 0.0f, { 0, 0, 0 } };
    return m;
}

static Material textured(Material m, RtTexture texture, float scale,
                         float strength, V3 alternate) {
    m.texture = texture;
    m.texture_scale = scale;
    m.texture_strength = strength;
    m.texture_color = alternate;
    return m;
}

static Material emit_mat(float r, float g, float b, float power) {
    Material m = mat(r, g, b, 0.25f, 48.0f);
    m.emission = v3(r * power, g * power, b * power);
    return m;
}

static void add_light(Scene *s, V3 position, V3 color, float power,
                      float radius, int cast_shadows) {
    if (s->light_count >= RT_MAX_LIGHTS) return;
    s->lights[s->light_count++] = (RtPointLight){
        position, color, power, radius, cast_shadows, 0.0f
    };
}

/* the original single-block verification scene */
void scene_block(Scene *s) {
    memset(s, 0, sizeof *s);
    add_plane(s, 0.0f, 1, mat(0.82f, 0.78f, 0.70f, 0.0f, 32.0f));
    add_box(s, v3(-1.2f, 0.0f, -1.2f), v3(1.2f, 2.4f, 1.2f),
            mat(0.85f, 0.35f, 0.20f, 0.25f, 48.0f));
    add_sphere(s, v3(2.35f, 0.8f, 1.7f), 0.8f,
               mat(0.30f, 0.50f, 0.85f, 0.35f, 64.0f));
    s->camera = v3(5.4f, 3.6f, 6.2f);
    s->target = v3(0.0f, 1.2f, 0.0f);
    s->fov = 50.0f;
    s->light = v3(4.2f, 6.5f, 3.2f);
    s->light_color = v3(1.0f, 0.95f, 0.85f);
    s->light_power = 100.0f;
    s->light_radius = 0.45f;
    add_light(s, s->light, s->light_color, s->light_power, s->light_radius, 1);
    s->lights[0].temperature = 6500.0f;
    rt_build_accel(s);
}

/* ---------------- miniature world diorama ---------------- */

static void tree(Scene *s, float x, float z, float scale) {
    Material trunk = mat(0.38f, 0.26f, 0.16f, 0.0f, 8.0f);
    Material leaf1 = mat(0.16f, 0.45f, 0.20f, 0.0f, 12.0f);
    Material leaf2 = mat(0.22f, 0.55f, 0.24f, 0.0f, 12.0f);
    add_cyl(s, v3(x, 0.0f, z), 0.22f * scale, 1.6f * scale, trunk);
    float cy = 1.6f * scale;
    add_sphere(s, v3(x, cy + 0.55f * scale, z), 0.85f * scale, leaf1);
    add_sphere(s, v3(x - 0.5f * scale, cy + 0.25f * scale, z + 0.3f * scale),
               0.6f * scale, leaf2);
    add_sphere(s, v3(x + 0.5f * scale, cy + 0.30f * scale, z - 0.25f * scale),
               0.62f * scale, leaf2);
    add_sphere(s, v3(x, cy + 1.15f * scale, z), 0.55f * scale, leaf1);
}

static void house(Scene *s, float x, float z, float w, float d, float h,
                  float wallR, float wallG, float wallB, float roofR,
                  float roofG, float roofB) {
    Material wall = mat(wallR, wallG, wallB, 0.0f, 16.0f);
    Material roof = mat(roofR, roofG, roofB, 0.0f, 16.0f);
    Material trim = mat(0.95f, 0.92f, 0.85f, 0.0f, 24.0f);
    Material window = mat(0.35f, 0.65f, 0.75f, 0.40f, 90.0f);
    float x0 = x - w * 0.5f, x1 = x + w * 0.5f;
    float z0 = z - d * 0.5f, z1 = z + d * 0.5f;
    /* walls */
    add_box(s, v3(x0, 0, z0), v3(x1, h, z1), wall);
    /* flat roof slab overhanging slightly */
    add_box(s, v3(x0 - 0.12f, h, z0 - 0.12f), v3(x1 + 0.12f, h + 0.22f, z1 + 0.12f), roof);
    /* door (front face, +z side) */
    add_box(s, v3(x - 0.25f, 0, z1 + 0.01f), v3(x + 0.25f, 1.1f, z1 + 0.06f), trim);
    /* windows: front and sides */
    add_box(s, v3(x0 + 0.25f, 1.2f, z1 + 0.01f), v3(x0 + 0.75f, 1.8f, z1 + 0.06f), window);
    add_box(s, v3(x1 - 0.75f, 1.2f, z1 + 0.01f), v3(x1 - 0.25f, 1.8f, z1 + 0.06f), window);
    add_box(s, v3(x1 + 0.01f, 1.2f, z - 0.25f), v3(x1 + 0.06f, 1.8f, z + 0.25f), window);
}

void scene_world(Scene *s) {
    memset(s, 0, sizeof *s);

    /* ground */
    add_plane(s, 0.0f, 0, mat(0.55f, 0.58f, 0.62f, 0.0f, 8.0f));

    /* grass island: two stacked slabs (dirt base, grass top) */
    Material grass = mat(0.30f, 0.62f, 0.28f, 0.0f, 8.0f);
    Material dirt = mat(0.42f, 0.30f, 0.20f, 0.0f, 8.0f);
    Material stone = mat(0.68f, 0.66f, 0.62f, 0.0f, 12.0f);
    Material path = mat(0.76f, 0.72f, 0.64f, 0.0f, 10.0f);
    add_box(s, v3(-9.0f, 0.0f, -8.0f), v3(9.0f, 0.4f, 8.0f), dirt);
    add_box(s, v3(-8.8f, 0.4f, -7.8f), v3(8.8f, 0.7f, 7.8f), grass);

    /* stone plaza + cross paths on top (y = 0.7 .. 0.76) */
    add_box(s, v3(-2.6f, 0.70f, -2.6f), v3(2.6f, 0.76f, 2.6f), path);
    add_box(s, v3(-8.2f, 0.70f, -0.9f), v3(8.2f, 0.76f, 0.9f), path);
    add_box(s, v3(-0.9f, 0.70f, -7.2f), v3(0.9f, 0.76f, 7.2f), path);
    /* stone well in plaza center */
    add_cyl(s, v3(0.0f, 0.76f, 0.0f), 0.7f, 0.6f, stone);
    add_cyl(s, v3(0.0f, 0.76f, 0.0f), 0.55f, 0.68f, dirt);

    /* houses: pastel + terracotta palette */
    house(s, -5.2f, -4.6f, 2.6f, 2.2f, 2.0f, 0.93f, 0.80f, 0.62f, 0.62f, 0.28f, 0.18f);
    house(s, 4.9f, -4.8f, 2.4f, 2.0f, 2.4f, 0.85f, 0.55f, 0.45f, 0.45f, 0.22f, 0.15f);
    house(s, 5.6f, 3.6f, 2.8f, 2.4f, 1.8f, 0.72f, 0.80f, 0.78f, 0.50f, 0.30f, 0.20f);
    house(s, -5.4f, 3.8f, 2.2f, 2.0f, 2.2f, 0.90f, 0.72f, 0.55f, 0.55f, 0.25f, 0.16f);

    /* trees (cylinder trunks + sphere foliage clusters) */
    tree(s, -7.4f, -6.6f, 1.15f);
    tree(s, 7.6f, -2.2f, 0.95f);
    tree(s, 1.8f, 6.2f, 1.05f);
    tree(s, -2.0f, 6.6f, 0.85f);
    tree(s, 7.8f, 6.9f, 1.2f);
    tree(s, -7.8f, 0.5f, 0.9f);

    /* crate stack near plaza */
    Material crate = mat(0.72f, 0.55f, 0.32f, 0.0f, 10.0f);
    add_box(s, v3(2.2f, 0.76f, 2.2f), v3(3.1f, 1.66f, 3.1f), crate);
    add_box(s, v3(2.3f, 1.66f, 2.3f), v3(3.0f, 2.41f, 3.0f), crate);

    /* lamp post: cylinder + golden sphere */
    Material lamp = mat(0.25f, 0.26f, 0.28f, 0.0f, 40.0f);
    Material gold = mat(0.95f, 0.78f, 0.30f, 0.55f, 90.0f);
    add_cyl(s, v3(-1.2f, 0.76f, -3.4f), 0.07f, 2.6f, lamp);
    add_sphere(s, v3(-1.2f, 3.5f, -3.4f), 0.22f, gold);

    /* Mirror panels: thin vertical slabs placed around the plaza. */
    Material mirror = mat(0.92f, 0.95f, 1.0f, 0.94f, 180.0f);
    add_box(s, v3(-7.9f, 0.78f, 1.35f), v3(-7.82f, 3.2f, 4.1f), mirror);
    add_box(s, v3(6.8f, 0.78f, -1.75f), v3(9.0f, 3.2f, -1.67f), mirror);

    /* Portable colored light balls. Their object indices are exposed so the
       interactive viewer can push them and update the matching lights. */
    Material red_ball = emit_mat(1.0f, 0.025f, 0.015f, 4.5f);
    Material blue_ball = emit_mat(0.0f, 0.0f, 1.0f, 4.5f);
    s->red_ball_object = add_sphere(s, v3(-3.2f, 1.15f, 1.8f), 0.42f, red_ball);
    s->blue_ball_object = add_sphere(s, v3(3.0f, 1.15f, 1.8f), 0.42f, blue_ball);

    s->camera = v3(24.0f, 10.0f, 27.0f);
    s->target = v3(0.0f, 2.0f, 0.0f);
    s->fov = 46.0f;
    s->light = v3(-6.0f, 12.0f, 5.0f);
    s->light_color = v3(1.0f, 1.0f, 1.0f);
    s->light_power = 230.0f;
    s->light_radius = 0.9f;
    add_light(s, s->light, s->light_color, s->light_power, s->light_radius, 1);
    s->lights[0].temperature = 6500.0f;
    add_light(s, v3(-3.2f, 1.15f, 1.8f), v3(1.0f, 0.0f, 0.0f), 110.0f, 0.2f, 0);
    add_light(s, v3(3.0f, 1.15f, 1.8f), v3(0.0f, 0.0f, 1.0f), 110.0f, 0.2f, 0);
    rt_build_accel(s);
}

/* Interactive scene: deliberately small enough for true 1x1 CPU rendering.
   The full 70-object diorama remains available through scene_world(). */
void scene_small(Scene *s) {
    memset(s, 0, sizeof *s);

    Material ground = textured(mat(0.48f, 0.52f, 0.48f, 0.0f, 8.0f), RT_TEX_CHECKER, 1.0f, .65f, v3(.24f, .27f, .25f));
    Material dirt = textured(mat(0.35f, 0.22f, 0.12f, 0.0f, 8.0f), RT_TEX_DIRT, 3.0f, .75f, v3(.20f, .11f, .055f));
    Material grass = textured(mat(0.22f, 0.48f, 0.18f, 0.0f, 8.0f), RT_TEX_GRASS, 5.0f, .70f, v3(.10f, .25f, .07f));
    Material stone = textured(mat(0.62f, 0.65f, 0.68f, 0.0f, 18.0f), RT_TEX_STONE, 2.4f, .55f, v3(.30f, .33f, .36f));
    Material wall = textured(mat(0.78f, 0.48f, 0.30f, 0.0f, 22.0f), RT_TEX_WALL, 2.0f, .25f, v3(.48f, .26f, .16f));
    Material wall2 = textured(mat(0.32f, 0.54f, 0.70f, 0.0f, 22.0f), RT_TEX_WALL, 2.0f, .22f, v3(.18f, .34f, .48f));
    Material roof = textured(mat(0.42f, 0.14f, 0.10f, 0.0f, 24.0f), RT_TEX_ROOF, 2.2f, .80f, v3(.12f, .035f, .025f));
    Material mirror = mat(0.94f, 0.96f, 1.0f, 0.96f, 180.0f);

    add_plane(s, 0.0f, 1, ground);
    add_box(s, v3(-8.0f, 0.0f, -7.0f), v3(8.0f, 0.45f, 7.0f), dirt);
    add_box(s, v3(-7.7f, 0.45f, -6.7f), v3(7.7f, 0.70f, 6.7f), grass);
    add_box(s, v3(-2.5f, 0.70f, -2.5f), v3(2.5f, 0.76f, 2.5f), stone);
    add_box(s, v3(-0.65f, 0.76f, -0.65f), v3(0.65f, 1.28f, 0.65f), stone);

    /* Two compact houses across the plaza. */
    add_box(s, v3(-5.8f, 0.76f, -4.8f), v3(-3.1f, 2.55f, -2.5f), wall);
    add_box(s, v3(-6.0f, 2.55f, -5.0f), v3(-2.9f, 2.82f, -2.3f), roof);
    add_box(s, v3(3.0f, 0.76f, 2.4f), v3(5.8f, 2.35f, 4.7f), wall2);
    add_box(s, v3(2.8f, 2.35f, 2.2f), v3(6.0f, 2.62f, 4.9f), roof);

    /* Mirrors face the center so they remain visible from the spawn point. */
    add_box(s, v3(-7.0f, 0.78f, 0.9f), v3(-6.92f, 2.9f, 3.0f), mirror);
    add_box(s, v3(6.5f, 0.78f, -1.0f), v3(6.58f, 2.9f, 1.1f), mirror);

    Material red = emit_mat(1.0f, 0.0f, 0.0f, 4.5f);
    Material blue = emit_mat(0.0f, 0.0f, 1.0f, 4.5f);
    s->red_ball_object = add_sphere(s, v3(-2.2f, 1.22f, 0.8f), 0.42f, red);
    s->blue_ball_object = add_sphere(s, v3(2.2f, 1.22f, 0.8f), 0.42f, blue);
    Material bunny = mat(.62f, .66f, .70f, .18f, 64.0f);
    int bunny_object = rt_load_mesh(s, "../assets/models/bunny/reconstruction/bun_zipper_res4.ply",
                                    v3(0.0f, 0.78f, -0.65f), 22.0f, bunny);
    if (bunny_object < 0) {
        bunny_object = rt_load_mesh(s, "assets/models/bunny/reconstruction/bun_zipper_res4.ply",
                                    v3(0.0f, 0.78f, -0.65f), 22.0f, bunny);
    }
    fprintf(stderr, "scene_small: bunny mesh %s (%d vertices, %d triangles)\n",
            bunny_object >= 0 ? "loaded" : "not found",
            bunny_object >= 0 ? s->meshes[s->objects[bunny_object].geometry.mesh.mesh].vertex_count : 0,
            bunny_object >= 0 ? s->meshes[s->objects[bunny_object].geometry.mesh.mesh].triangle_count : 0);

    s->camera = v3(0.0f, 3.4f, 7.5f);
    s->target = v3(0.0f, 1.6f, 0.0f);
    s->fov = 72.0f;
    s->light = v3(-3.0f, 10.0f, 4.0f);
    s->light_color = v3(1.0f, 1.0f, 1.0f);
    s->light_power = 180.0f;
    s->light_radius = 0.0f;
    add_light(s, s->light, s->light_color, s->light_power, s->light_radius, 0);
    s->lights[0].temperature = 6500.0f;
    add_light(s, v3(-2.2f, 1.22f, 0.8f), v3(1.0f, 0.0f, 0.0f), 75.0f, 0.0f, 0);
    add_light(s, v3(2.2f, 1.22f, 0.8f), v3(0.0f, 0.0f, 1.0f), 75.0f, 0.0f, 0);
    rt_build_accel(s);
}
