/* Render Studio - engine editor shell (clay v0.14). VISUAL ONLY:
 * every panel shows mock data; nothing is wired to the tracer yet. */
#define CLAY_IMPLEMENTATION
#include "../../../clay/clay.h"
#include "../../../clay/clay_renderer_raylib.c"

#include <string.h>
#include <stdlib.h>

#define WINDOW_W 1024
#define WINDOW_H 640

static const uint32_t F_BODY = 0;
static const uint32_t F_MONO = 1;

/* palette */
static const Clay_Color BG_DARK     = { 24, 26, 32, 255 };
static const Clay_Color BG_PANEL    = { 32, 35, 44, 255 };
static const Clay_Color BG_CONTROL  = { 44, 48, 60, 255 };
static const Clay_Color BG_HOVER    = { 56, 62, 78, 255 };
static const Clay_Color TXT_MAIN    = { 235, 238, 245, 255 };
static const Clay_Color TXT_DIM     = { 140, 148, 165, 255 };
static const Clay_Color ACCENT      = { 225, 138, 50, 255 };
static const Clay_Color ACCENT_DIM  = { 140, 90, 40, 255 };
static const Clay_Color OUTLINE     = { 60, 66, 82, 255 };
static const Clay_Color VIEW_BG     = { 16, 17, 22, 255 };
static const Clay_Color OVERLAY_BG  = { 0, 0, 0, 140 };
static const Clay_Color GREEN_OK    = { 140, 200, 120, 255 };
static const Clay_Color YELLOW_WARN = { 230, 200, 110, 255 };
static const Clay_Color RED_ERR     = { 220, 110, 100, 255 };

/* ---- mock state (visual only) ---- */
static int selTool = 1;       /* 0 select, 1 move, 2 rotate, 3 scale */
static bool playing = true;
static int selTreeRow = 6;    /* House_A */
static int dockTab = 0;       /* 0 assets, 1 console, 2 timeline */

static void ClayErrorHandler(Clay_ErrorData errorData) {
    (void)errorData;
}

static Clay_String S(const char *s) {
    Clay_String str = { .isStaticallyAllocated = true, .length = (int32_t)strlen(s), .chars = s };
    return str;
}

static Color RayColor(Clay_Color c) {
    return (Color){ (unsigned char)c.r, (unsigned char)c.g, (unsigned char)c.b, (unsigned char)c.a };
}

static Clay_TextElementConfig txtMenu  = { .fontId = 0, .fontSize = 13, .textColor = { 200, 205, 215, 255 } };
static Clay_TextElementConfig txtTree  = { .fontId = 0, .fontSize = 13, .textColor = { 200, 205, 215, 255 } };
static Clay_TextElementConfig txtSm    = { .fontId = 0, .fontSize = 12, .textColor = { 140, 148, 165, 255 } };
static Clay_TextElementConfig txtH     = { .fontId = 0, .fontSize = 12, .letterSpacing = 2, .textColor = { 225, 138, 50, 255 } };
static Clay_TextElementConfig txtVal   = { .fontId = 1, .fontSize = 12, .textColor = { 235, 238, 245, 255 } };
static Clay_TextElementConfig txtMono  = { .fontId = 1, .fontSize = 12, .textColor = { 140, 148, 165, 255 } };
static Clay_TextElementConfig txtTool  = { .fontId = 0, .fontSize = 13, .textColor = { 235, 238, 245, 255 } };
static Clay_TextElementConfig txtTitle = { .fontId = 0, .fontSize = 16, .textColor = { 235, 238, 245, 255 } };

/* ---- hierarchy mock data ---- */
enum { T_FOLDER, T_LEAF };
typedef struct { int depth; const char *label; int kind; } TreeRowData;
static const TreeRowData treeData[] = {
    { 0, "MiniWorld.scene", T_FOLDER },
    { 1, "terrain_island",  T_LEAF   },
    { 1, "plaza",           T_FOLDER },
    { 2, "well",            T_LEAF   },
    { 2, "path_stones",     T_LEAF   },
    { 1, "houses",          T_FOLDER },
    { 2, "House_A",         T_LEAF   },
    { 2, "House_B",         T_LEAF   },
    { 2, "House_C",         T_LEAF   },
    { 2, "House_D",         T_LEAF   },
    { 1, "trees",           T_FOLDER },
    { 2, "Tree_A",          T_LEAF   },
    { 2, "Tree_B",          T_LEAF   },
    { 2, "Tree_C",          T_LEAF   },
    { 2, "Tree_D",          T_LEAF   },
    { 2, "Tree_E",          T_LEAF   },
    { 2, "Tree_F",          T_LEAF   },
    { 1, "props",           T_FOLDER },
    { 2, "crates",          T_LEAF   },
    { 2, "lamp_post",       T_LEAF   },
    { 2, "gold_sphere",     T_LEAF   },
};
static const int treeN = (int)(sizeof(treeData) / sizeof(treeData[0]));

/* ---- assets mock data ---- */
typedef struct { const char *name; Clay_Color color; } AssetItem;
static const AssetItem assetData[] = {
    { "house_a.obj",    { 224, 122,  80, 255 } },
    { "pine.fbx",       {  90, 140,  90, 255 } },
    { "brick.mat",      { 150,  90,  60, 255 } },
    { "well.stone",     { 120, 125, 135, 255 } },
    { "lamp.obj",       { 200, 170,  80, 255 } },
    { "sky.hdri",       { 100, 150, 220, 255 } },
    { "crate.fbx",      { 140, 110,  70, 255 } },
    { "gold.mat",       { 210, 180,  90, 255 } },
    { "roof.mat",       { 170,  80,  60, 255 } },
    { "grass.mat",      {  90, 150,  80, 255 } },
    { "door.fbx",       { 110,  85,  60, 255 } },
    { "path_stone.mat", { 135, 135, 140, 255 } },
};
static const int assetN = (int)(sizeof(assetData) / sizeof(assetData[0]));

/* ================= helpers ================= */

static void MenuItem(int i, const char *label) {
    Clay_ElementDeclaration d = {
        .layout = { .sizing = { .width = CLAY_SIZING_FIT(), .height = CLAY_SIZING_FIXED(26) },
                    .padding = { 10, 10, 0, 0 }, .childAlignment = { .y = CLAY_ALIGN_Y_CENTER } },
    };
    d.id = CLAY_IDI("mb", i);
    if (Clay_Hovered()) d.backgroundColor = BG_HOVER;
    CLAY(d) {
        CLAY_TEXT(S(label), CLAY_TEXT_CONFIG(txtMenu));
    }
}

static void ToolBtn(int i, const char *glyph, bool selected) {
    Clay_ElementDeclaration d = {
        .layout = { .sizing = { .width = CLAY_SIZING_FIXED(28), .height = CLAY_SIZING_FIXED(28) },
                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER } },
        .backgroundColor = selected ? ACCENT : BG_CONTROL,
        .cornerRadius = CLAY_CORNER_RADIUS(5),
        .border = { .width = { 1, 1, 1, 1 }, .color = selected ? ACCENT : OUTLINE },
    };
    d.id = CLAY_IDI("tb", i);
    CLAY(d) {
        if (selected) {
            Clay_TextElementConfig cfg = { .fontId = 0, .fontSize = 13, .textColor = { 25, 20, 12, 255 } };
            CLAY_TEXT(S(glyph), CLAY_TEXT_CONFIG(cfg));
        } else {
            CLAY_TEXT(S(glyph), CLAY_TEXT_CONFIG(txtTool));
        }
    }
}

static void PlayBtn(int i, const char *label, bool selected) {
    Clay_ElementDeclaration d = {
        .layout = { .sizing = { .width = CLAY_SIZING_FIT(), .height = CLAY_SIZING_FIXED(24) },
                    .padding = { 12, 12, 0, 0 }, .childAlignment = { .y = CLAY_ALIGN_Y_CENTER } },
        .backgroundColor = selected ? ACCENT : BG_CONTROL,
        .cornerRadius = CLAY_CORNER_RADIUS(5),
        .border = { .width = { 1, 1, 1, 1 }, .color = selected ? ACCENT : OUTLINE },
    };
    d.id = CLAY_IDI("pb", i);
    CLAY(d) {
        if (selected) {
            Clay_TextElementConfig cfg = { .fontId = 0, .fontSize = 12, .letterSpacing = 1, .textColor = { 25, 20, 12, 255 } };
            CLAY_TEXT(S(label), CLAY_TEXT_CONFIG(cfg));
        } else {
            Clay_TextElementConfig cfg = { .fontId = 0, .fontSize = 12, .letterSpacing = 1, .textColor = { 200, 205, 215, 255 } };
            CLAY_TEXT(S(label), CLAY_TEXT_CONFIG(cfg));
        }
    }
}

static void TreeRow(int i, const TreeRowData *row) {
    bool selected = (i == selTreeRow);
    Clay_ElementDeclaration d = {
        .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(24) },
                    .padding = { (uint16_t)(10 + row->depth * 16), 8, 0, 0 },
                    .childAlignment = { .y = CLAY_ALIGN_Y_CENTER }, .childGap = 8 },
        .backgroundColor = selected ? ACCENT_DIM : (Clay_Hovered() ? BG_HOVER : BG_DARK),
        .border = { .width = { selected ? 2u : 0u, 0, 0, 0 }, .color = ACCENT },
    };
    d.id = CLAY_IDI("hrow", i);
    CLAY(d) {
        if (row->kind == T_FOLDER) {
            CLAY_TEXT(S("v"), CLAY_TEXT_CONFIG(txtSm));
        } else {
            CLAY({
                .id = CLAY_IDI("hdot", i),
                .layout = { .sizing = { .width = CLAY_SIZING_FIXED(6), .height = CLAY_SIZING_FIXED(6) } },
                .backgroundColor = OUTLINE,
                .cornerRadius = CLAY_CORNER_RADIUS(2),
            }) {}
        }
        CLAY_TEXT(S(row->label), CLAY_TEXT_CONFIG(txtTree));
    }
}

static void SectionHeader(int i, const char *text) {
    CLAY({
        .id = CLAY_IDI("sect", i),
        .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(22) },
                    .childAlignment = { .y = CLAY_ALIGN_Y_CENTER } },
        .border = { .width = { 0, 0, 0, 1 }, .color = OUTLINE },
    }) {
        CLAY_TEXT(S(text), CLAY_TEXT_CONFIG(txtH));
    }
}

/* label + three value boxes (transform rows) */
static void FieldRow3(int i, const char *label, const char *vx, const char *vy, const char *vz) {
    CLAY({
        .id = CLAY_IDI("frow", i),
        .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(24) },
                    .childAlignment = { .y = CLAY_ALIGN_Y_CENTER }, .childGap = 6 },
    }) {
        CLAY({
            .id = CLAY_IDI("flab", i),
            .layout = { .sizing = { .width = CLAY_SIZING_FIXED(70), .height = CLAY_SIZING_GROW(0) },
                        .childAlignment = { .y = CLAY_ALIGN_Y_CENTER } },
        }) {
            CLAY_TEXT(S(label), CLAY_TEXT_CONFIG(txtSm));
        }
        const char *vals[3] = { vx, vy, vz };
        for (int k = 0; k < 3; k++) {
            CLAY({
                .id = CLAY_IDI("fld", i * 10 + k),
                .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(22) },
                            .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER } },
                .backgroundColor = BG_CONTROL,
                .cornerRadius = CLAY_CORNER_RADIUS(4),
            }) {
                CLAY_TEXT(S(vals[k]), CLAY_TEXT_CONFIG(txtVal));
            }
        }
    }
}

/* label + slider track + value */
static void SliderRow(int i, const char *label, float frac, const char *val) {
    CLAY({
        .id = CLAY_IDI("srow", i),
        .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(24) },
                    .childAlignment = { .y = CLAY_ALIGN_Y_CENTER }, .childGap = 6 },
    }) {
        CLAY({
            .id = CLAY_IDI("slab", i),
            .layout = { .sizing = { .width = CLAY_SIZING_FIXED(70), .height = CLAY_SIZING_GROW(0) },
                        .childAlignment = { .y = CLAY_ALIGN_Y_CENTER } },
        }) {
            CLAY_TEXT(S(label), CLAY_TEXT_CONFIG(txtSm));
        }
        CLAY({
            .id = CLAY_IDI("strk", i),
            .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(12) } },
            .backgroundColor = BG_CONTROL,
            .cornerRadius = CLAY_CORNER_RADIUS(6),
        }) {
            CLAY({
                .id = CLAY_IDI("sfil", i),
                .layout = { .sizing = { .width = CLAY_SIZING_PERCENT(frac), .height = CLAY_SIZING_GROW(0) } },
                .backgroundColor = ACCENT,
                .cornerRadius = CLAY_CORNER_RADIUS(6),
            }) {}
        }
        CLAY_TEXT(S(val), CLAY_TEXT_CONFIG(txtVal));
    }
}

/* label + checkbox + state text */
static void CheckRow(int i, const char *label, bool on, const char *state) {
    CLAY({
        .id = CLAY_IDI("crow", i),
        .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(24) },
                    .childAlignment = { .y = CLAY_ALIGN_Y_CENTER }, .childGap = 6 },
    }) {
        CLAY({
            .id = CLAY_IDI("clab", i),
            .layout = { .sizing = { .width = CLAY_SIZING_FIXED(70), .height = CLAY_SIZING_GROW(0) },
                        .childAlignment = { .y = CLAY_ALIGN_Y_CENTER } },
        }) {
            CLAY_TEXT(S(label), CLAY_TEXT_CONFIG(txtSm));
        }
        CLAY({
            .id = CLAY_IDI("cbox", i),
            .layout = { .sizing = { .width = CLAY_SIZING_FIXED(14), .height = CLAY_SIZING_FIXED(14) },
                        .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER } },
            .backgroundColor = BG_CONTROL,
            .cornerRadius = CLAY_CORNER_RADIUS(3),
            .border = { .width = { 1, 1, 1, 1 }, .color = OUTLINE },
        }) {
            if (on) {
                CLAY({
                    .id = CLAY_IDI("cfill", i),
                    .layout = { .sizing = { .width = CLAY_SIZING_FIXED(8), .height = CLAY_SIZING_FIXED(8) } },
                    .backgroundColor = ACCENT,
                    .cornerRadius = CLAY_CORNER_RADIUS(2),
                }) {}
            }
        }
        CLAY_TEXT(S(state), CLAY_TEXT_CONFIG(txtMono));
    }
}

static void AssetTile(int i, const AssetItem *a) {
    CLAY({
        .id = CLAY_IDI("atile", i),
        .layout = { .layoutDirection = CLAY_TOP_TO_BOTTOM, .childGap = 4,
                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER } },
    }) {
        CLAY({
            .id = CLAY_IDI("athumb", i),
            .layout = { .sizing = { .width = CLAY_SIZING_FIXED(64), .height = CLAY_SIZING_FIXED(64) } },
            .backgroundColor = a->color,
            .cornerRadius = CLAY_CORNER_RADIUS(6),
            .border = { .width = { 1, 1, 1, 1 }, .color = OUTLINE },
        }) {}
        CLAY_TEXT(S(a->name), CLAY_TEXT_CONFIG(txtSm));
    }
}

static void ConsoleLine(int i, const char *time, const char *lvl, Clay_Color lvlColor, const char *msg) {
    CLAY({
        .id = CLAY_IDI("cline", i),
        .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(20) },
                    .childAlignment = { .y = CLAY_ALIGN_Y_CENTER }, .childGap = 10 },
    }) {
        CLAY_TEXT(S(time), CLAY_TEXT_CONFIG(txtMono));
        Clay_TextElementConfig lvlCfg = { .fontId = 1, .fontSize = 12, .textColor = lvlColor };
        CLAY_TEXT(S(lvl), CLAY_TEXT_CONFIG(lvlCfg));
        CLAY_TEXT(S(msg), CLAY_TEXT_CONFIG(txtMono));
    }
}

/* ================= layout ================= */

static const char *timelineTicks[7] = { "0", "8", "16", "24", "32", "40", "48" };

static Clay_RenderCommandArray CreateLayout(void) {
    Clay_BeginLayout();

    CLAY({
        .id = CLAY_ID("Root"),
        .layout = { .layoutDirection = CLAY_TOP_TO_BOTTOM,
                    .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0) },
                    .childGap = 0 },
        .backgroundColor = BG_DARK,
    }) {
        /* ---- menu bar ---- */
        CLAY({
            .id = CLAY_ID("MenuBar"),
            .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(26) },
                        .padding = { 6, 6, 0, 0 }, .childGap = 0 },
            .backgroundColor = BG_PANEL,
            .border = { .width = { 0, 0, 1, 0 }, .color = OUTLINE },
        }) {
            MenuItem(0, "File"); MenuItem(1, "Edit"); MenuItem(2, "Assets");
            MenuItem(3, "Component"); MenuItem(4, "Window"); MenuItem(5, "Help");
        }

        /* ---- toolbar ---- */
        CLAY({
            .id = CLAY_ID("ToolBar"),
            .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(36) },
                        .padding = { 8, 8, 0, 0 }, .childAlignment = { .y = CLAY_ALIGN_Y_CENTER }, .childGap = 4 },
            .backgroundColor = BG_PANEL,
            .border = { .width = { 0, 0, 1, 0 }, .color = OUTLINE },
        }) {
            ToolBtn(0, "Q", selTool == 0);
            ToolBtn(1, "W", selTool == 1);
            ToolBtn(2, "E", selTool == 2);
            ToolBtn(3, "R", selTool == 3);
            CLAY({
                .id = CLAY_ID("ToolSep"),
                .layout = { .sizing = { .width = CLAY_SIZING_FIXED(1), .height = CLAY_SIZING_FIXED(20) } },
                .backgroundColor = OUTLINE,
            }) {}
            CLAY_TEXT(S("House_A"), CLAY_TEXT_CONFIG(txtSm));
            CLAY({
                .id = CLAY_ID("ToolSpacer1"),
                .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0) } },
            }) {}
            PlayBtn(0, "PLAY", playing);
            PlayBtn(1, "PAUSE", false);
            PlayBtn(2, "STOP", false);
            CLAY({
                .id = CLAY_ID("ToolSpacer2"),
                .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0) } },
            }) {}
            CLAY({
                .id = CLAY_ID("CamMode"),
                .layout = { .sizing = { .width = CLAY_SIZING_FIT(), .height = CLAY_SIZING_FIXED(24) },
                            .padding = { 12, 12, 0, 0 }, .childAlignment = { .y = CLAY_ALIGN_Y_CENTER } },
                .backgroundColor = BG_CONTROL,
                .cornerRadius = CLAY_CORNER_RADIUS(5),
            }) {
                CLAY_TEXT(S("Perspective"), CLAY_TEXT_CONFIG(txtSm));
            }
        }

        /* ---- body: hierarchy | center | inspector ---- */
        CLAY({
            .id = CLAY_ID("Body"),
            .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0) }, .childGap = 0 },
        }) {
            /* ===== hierarchy ===== */
            CLAY({
                .id = CLAY_ID("Hierarchy"),
                .layout = { .layoutDirection = CLAY_TOP_TO_BOTTOM,
                            .sizing = { .width = CLAY_SIZING_FIXED(240), .height = CLAY_SIZING_GROW(0) },
                            .padding = { 8, 8, 8, 4 }, .childGap = 6 },
                .backgroundColor = BG_DARK,
                .border = { .width = { 0, 1, 0, 0 }, .color = OUTLINE },
            }) {
                SectionHeader(0, "HIERARCHY");
                CLAY({
                    .id = CLAY_ID("TreeScroll"),
                    .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0) } },
                    .clip = { .vertical = true, .childOffset = Clay_GetScrollOffset() },
                }) {
                    CLAY({
                        .id = CLAY_ID("TreeRows"),
                        .layout = { .layoutDirection = CLAY_TOP_TO_BOTTOM, .childGap = 1,
                                    .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIT() } },
                    }) {
                        for (int i = 0; i < treeN; i++) {
                            TreeRow(i, &treeData[i]);
                        }
                    }
                }
            }

            /* ===== center column ===== */
            CLAY({
                .id = CLAY_ID("Center"),
                .layout = { .layoutDirection = CLAY_TOP_TO_BOTTOM,
                            .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0) },
                            .childGap = 0 },
            }) {
                /* viewport */
                CLAY({
                    .id = CLAY_ID("Viewport"),
                    .layout = { .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0) } },
                    .backgroundColor = VIEW_BG,
                }) {
                    CLAY({
                        .id = CLAY_ID("ViewTop"),
                        .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(24) },
                                    .padding = { 10, 10, 0, 0 }, .childAlignment = { .y = CLAY_ALIGN_Y_CENTER } },
                        .backgroundColor = OVERLAY_BG,
                    }) {
                        CLAY_TEXT(S("Perspective  |  MiniWorld.scene"), CLAY_TEXT_CONFIG(txtSm));
                    }
                    CLAY({
                        .id = CLAY_ID("ViewMid"),
                        .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0) },
                                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER } },
                    }) {
                        CLAY({
                            .id = CLAY_ID("ViewHint"),
                            .layout = { .layoutDirection = CLAY_TOP_TO_BOTTOM, .childGap = 8,
                                        .childAlignment = { .x = CLAY_ALIGN_X_CENTER } },
                        }) {
                            CLAY_TEXT(S("preview"), CLAY_TEXT_CONFIG({ .fontId = 0, .fontSize = 20, .letterSpacing = 2, .textColor = { 90, 98, 115, 255 } }));
                            CLAY_TEXT(S("render output appears here"), CLAY_TEXT_CONFIG({ .fontId = 1, .fontSize = 13, .textColor = { 70, 78, 95, 255 } }));
                        }
                    }
                    CLAY({
                        .id = CLAY_ID("ViewBottom"),
                        .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(24) },
                                    .padding = { 10, 10, 0, 0 },
                                    .childAlignment = { .x = CLAY_ALIGN_X_RIGHT, .y = CLAY_ALIGN_Y_CENTER } },
                        .backgroundColor = OVERLAY_BG,
                    }) {
                        CLAY_TEXT(S("1280 x 720  |  4 spp  |  38.2 s  |  cpu"), CLAY_TEXT_CONFIG(txtMono));
                    }
                }

                /* ---- bottom dock ---- */
                CLAY({
                    .id = CLAY_ID("Dock"),
                    .layout = { .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(190) },
                                .childGap = 0 },
                    .backgroundColor = BG_PANEL,
                    .border = { .width = { 0, 1, 0, 0 }, .color = OUTLINE },
                }) {
                    CLAY({
                        .id = CLAY_ID("TabBar"),
                        .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(28) },
                                    .padding = { 8, 8, 0, 0 }, .childGap = 2 },
                    }) {
                        const char *tabs[3] = { "ASSETS", "CONSOLE", "TIMELINE" };
                        for (int t = 0; t < 3; t++) {
                            bool sel = (dockTab == t);
                            Clay_ElementDeclaration d = {
                                .layout = { .sizing = { .width = CLAY_SIZING_FIT(), .height = CLAY_SIZING_GROW(0) },
                                            .padding = { 12, 12, 0, 0 },
                                            .childAlignment = { .y = CLAY_ALIGN_Y_CENTER } },
                                .border = { .width = { 0, 0, 0, sel ? 2u : 0u }, .color = ACCENT },
                            };
                            d.id = CLAY_IDI("tab", t);
                            CLAY(d) {
                                Clay_TextElementConfig cfg = { .fontId = 0, .fontSize = 12, .letterSpacing = 1,
                                                               .textColor = sel ? TXT_MAIN : TXT_DIM };
                                CLAY_TEXT(S(tabs[t]), CLAY_TEXT_CONFIG(cfg));
                            }
                        }
                    }
                    CLAY({
                        .id = CLAY_ID("TabContent"),
                        .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0) },
                                    .padding = { 10, 10, 6, 6 } },
                    }) {
                        if (dockTab == 0) {
                            /* assets: horizontal scroll strip */
                            CLAY({
                                .id = CLAY_ID("AssetScroll"),
                                .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0) } },
                                .clip = { .horizontal = true, .childOffset = Clay_GetScrollOffset() },
                            }) {
                                CLAY({
                                    .id = CLAY_ID("AssetRow"),
                                    .layout = { .childGap = 14, .sizing = { .height = CLAY_SIZING_FIT() } },
                                }) {
                                    for (int a = 0; a < assetN; a++) {
                                        AssetTile(a, &assetData[a]);
                                    }
                                }
                            }
                        } else if (dockTab == 1) {
                            ConsoleLine(0, "00:00:00", "OK",   GREEN_OK,    "scene_world: built 66 objects, 12 materials in 0.41 ms");
                            ConsoleLine(1, "00:00:00", "OK",   GREEN_OK,    "tracer: analytic intersections ready (box/sphere/cyl/plane)");
                            ConsoleLine(2, "00:00:01", "WARN", YELLOW_WARN, "soft shadows: 5-tap fixed offsets (deterministic mode)");
                            ConsoleLine(3, "00:00:02", "OK",   GREEN_OK,    "render 1280x720 @ 4 spp finished in 38.2 s (cpu)");
                            ConsoleLine(4, "00:00:02", "OK",   GREEN_OK,    "probe: sky 185,206,235 | terracotta block | floor lit");
                        } else {
                            /* timeline: ruler + tracks */
                            CLAY({
                                .id = CLAY_ID("TlRuler"),
                                .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(18) },
                                            .padding = { 0, 0, 0, 98 }, .childGap = 0 },
                            }) {
                                for (int f = 0; f <= 6; f++) {
                                    CLAY({
                                        .id = CLAY_IDI("tlr", f),
                                        .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0) },
                                                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER } },
                                    }) {
                                        CLAY_TEXT(S(timelineTicks[f]), CLAY_TEXT_CONFIG(txtMono));
                                    }
                                }
                            }
                            struct { const char *label; float start, len; bool active; } tracks[3] = {
                                { "House_A",     0.14f, 0.34f, true  },
                                { "lamp_post",   0.32f, 0.38f, false },
                                { "gold_sphere", 0.58f, 0.22f, false },
                            };
                            for (int t = 0; t < 3; t++) {
                                CLAY({
                                    .id = CLAY_IDI("tlt", t),
                                    .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(30) },
                                                .childAlignment = { .y = CLAY_ALIGN_Y_CENTER }, .childGap = 8 },
                                }) {
                                    CLAY({
                                        .id = CLAY_IDI("tlname", t),
                                        .layout = { .sizing = { .width = CLAY_SIZING_FIXED(90), .height = CLAY_SIZING_GROW(0) },
                                                    .childAlignment = { .y = CLAY_ALIGN_Y_CENTER } },
                                    }) {
                                        CLAY_TEXT(S(tracks[t].label), CLAY_TEXT_CONFIG(txtSm));
                                    }
                                    CLAY({
                                        .id = CLAY_IDI("tllane", t),
                                        .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(22) },
                                                    .padding = { 2, 2, 0, 0 } },
                                        .backgroundColor = BG_CONTROL,
                                        .cornerRadius = CLAY_CORNER_RADIUS(4),
                                    }) {
                                        CLAY({
                                            .id = CLAY_IDI("tlgap", t),
                                            .layout = { .sizing = { .width = CLAY_SIZING_PERCENT(tracks[t].start), .height = CLAY_SIZING_GROW(0) } },
                                        }) {}
                                        CLAY({
                                            .id = CLAY_IDI("tlclip", t),
                                            .layout = { .sizing = { .width = CLAY_SIZING_PERCENT(tracks[t].len), .height = CLAY_SIZING_GROW(0) },
                                                        .childAlignment = { .y = CLAY_ALIGN_Y_CENTER }, .padding = { 6, 2, 0, 0 } },
                                            .backgroundColor = tracks[t].active ? ACCENT_DIM : BG_HOVER,
                                            .cornerRadius = CLAY_CORNER_RADIUS(3),
                                            .border = { .width = { 1, 1, 1, 1 }, .color = tracks[t].active ? ACCENT : OUTLINE },
                                        }) {
                                            if (tracks[t].active) {
                                                Clay_TextElementConfig cfg = { .fontId = 0, .fontSize = 10, .textColor = { 25, 20, 12, 255 } };
                                                CLAY_TEXT(S("clip"), CLAY_TEXT_CONFIG(cfg));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            /* ===== inspector ===== */
            CLAY({
                .id = CLAY_ID("Inspector"),
                .layout = { .layoutDirection = CLAY_TOP_TO_BOTTOM,
                            .sizing = { .width = CLAY_SIZING_FIXED(300), .height = CLAY_SIZING_GROW(0) },
                            .padding = { 10, 10, 8, 6 }, .childGap = 6 },
                .backgroundColor = BG_DARK,
                .border = { .width = { 1, 0, 0, 0 }, .color = OUTLINE },
            }) {
                SectionHeader(1, "INSPECTOR");
                CLAY_TEXT(S("House_A"), CLAY_TEXT_CONFIG(txtTitle));
                CLAY_TEXT(S("RtObject #17  |  static mesh"), CLAY_TEXT_CONFIG(txtSm));

                SectionHeader(2, "TRANSFORM");
                FieldRow3(0, "Position", "-6.0", "0.0", "-3.0");
                FieldRow3(1, "Rotation", "0.0", "15.0", "0.0");
                FieldRow3(2, "Scale", "1.0", "1.0", "1.0");

                SectionHeader(3, "MATERIAL");
                CLAY({
                    .id = CLAY_ID("AlbedoRow"),
                    .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(26) },
                                .childAlignment = { .y = CLAY_ALIGN_Y_CENTER }, .childGap = 6 },
                }) {
                    CLAY({
                        .id = CLAY_ID("AlbedoLab"),
                        .layout = { .sizing = { .width = CLAY_SIZING_FIXED(70), .height = CLAY_SIZING_GROW(0) },
                                    .childAlignment = { .y = CLAY_ALIGN_Y_CENTER } },
                    }) {
                        CLAY_TEXT(S("Albedo"), CLAY_TEXT_CONFIG(txtSm));
                    }
                    CLAY({
                        .id = CLAY_ID("AlbedoSwatch"),
                        .layout = { .sizing = { .width = CLAY_SIZING_FIXED(44), .height = CLAY_SIZING_FIXED(20) } },
                        .backgroundColor = { 224, 122, 80, 255 },
                        .cornerRadius = CLAY_CORNER_RADIUS(4),
                        .border = { .width = { 1, 1, 1, 1 }, .color = OUTLINE },
                    }) {}
                    CLAY_TEXT(S("#E07A50"), CLAY_TEXT_CONFIG(txtMono));
                }
                SliderRow(0, "Reflection", 0.35f, "0.35");
                SliderRow(1, "Shininess", 0.40f, "32");

                SectionHeader(4, "RENDER");
                CheckRow(0, "Shadow", true, "on");
                CheckRow(1, "Visible", true, "on");
                CheckRow(2, "Bounces", false, "1 (global)");

                SectionHeader(5, "LIGHT");
                FieldRow3(3, "Color", "1.0", "0.92", "0.80");
                SliderRow(2, "Power", 0.76f, "230");
                SliderRow(3, "Radius", 0.30f, "0.9");
            }
        }

        /* ---- status bar ---- */
        CLAY({
            .id = CLAY_ID("StatusBar"),
            .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(24) },
                        .padding = { 10, 10, 0, 0 },
                        .childAlignment = { .y = CLAY_ALIGN_Y_CENTER }, .childGap = 16 },
            .backgroundColor = BG_PANEL,
            .border = { .width = { 0, 1, 0, 0 }, .color = OUTLINE },
        }) {
            CLAY_TEXT(S("MiniWorld.scene  -  66 objects  |  12 materials  |  selected: House_A"), CLAY_TEXT_CONFIG(txtSm));
            CLAY({
                .id = CLAY_ID("StatusSpacer"),
                .layout = { .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0) } },
            }) {}
            CLAY_TEXT(S("cpu ray tracer v0.2 (refactored)  |  clay v0.14  |  60 fps"), CLAY_TEXT_CONFIG(txtMono));
        }
    }

    return Clay_EndLayout();
}

int main(int argc, char **argv) {
    Clay_Raylib_Initialize(WINDOW_W, WINDOW_H, "Render Studio - clay", FLAG_VSYNC_HINT | FLAG_WINDOW_RESIZABLE);

    uint32_t minMem = Clay_MinMemorySize();
    void *mem = malloc(minMem);
    Clay_Arena arena = Clay_CreateArenaWithCapacityAndMemory(minMem, mem);
    Clay_Initialize(arena, (Clay_Dimensions){ (float)WINDOW_W, (float)WINDOW_H },
                    (Clay_ErrorHandler){ ClayErrorHandler, 0 });
    Font fonts[2];
    fonts[F_BODY] = LoadFontEx("Roboto-Regular.ttf", 40, 0, 400);
    SetTextureFilter(fonts[F_BODY].texture, TEXTURE_FILTER_BILINEAR);
    fonts[F_MONO] = LoadFontEx("Roboto-Regular.ttf", 32, 0, 400);
    SetTextureFilter(fonts[F_MONO].texture, TEXTURE_FILTER_BILINEAR);
    Clay_SetMeasureTextFunction(Raylib_MeasureText, fonts);

    /* NOTE: do not set FLAG_WINDOW_HIGHDPI - it makes TakeScreenshot and
     * friends report a DPI-scaled framebuffer that does not match the real
     * swapchain. Logical pixels == framebuffer pixels on this machine. */
    Clay_SetLayoutDimensions((Clay_Dimensions){ (float)WINDOW_W, (float)WINDOW_H });

    int frame = 0;
    while (!WindowShouldClose()) {
        Clay_SetPointerState((Clay_Vector2){ (float)GetMouseX(), (float)GetMouseY() },
                             IsMouseButtonDown(MOUSE_LEFT_BUTTON));
        Clay_SetLayoutDimensions((Clay_Dimensions){ (float)GetScreenWidth(), (float)GetScreenHeight() });
        Clay_UpdateScrollContainers(false, (Clay_Vector2){ 0, GetMouseWheelMove() * 30 }, GetFrameTime());

        BeginDrawing();
        ClearBackground(RayColor(BG_DARK));
        Clay_Raylib_Render(CreateLayout(), fonts);
        EndDrawing();

        if (argc > 1 && argv[1][0] == 'd') {
            if (++frame == 90) {
                TakeScreenshot("ui_dump.png");
                break;
            }
        }
    }

    Clay_Raylib_Close();
    return 0;
}
