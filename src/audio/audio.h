#ifndef VERLET_AUDIO_H
#define VERLET_AUDIO_H

#include "raylib.h"

#define VL_AUDIO_MAX_BUSES 16
#define VL_AUDIO_INVALID_ID (-1)

typedef enum { VL_AUDIO_BUS_MASTER = 0, VL_AUDIO_BUS_MUSIC, VL_AUDIO_BUS_SFX, VL_AUDIO_BUS_UI, VL_AUDIO_BUS_AMBIENCE } VlAudioBusId;

typedef struct { int id; char name[32]; float volume; int muted; } VlAudioBus;
typedef struct { Sound sound; int bus; float base_volume; int valid; } VlAudioSound;
typedef struct { Music music; int bus; float base_volume; int valid; } VlAudioMusic;
typedef struct { Wave wave; Sound sound; int valid; } VlAudioTone;

typedef struct {
    int initialized;
    VlAudioBus buses[VL_AUDIO_MAX_BUSES];
    int bus_count;
} VlAudio;

int vl_audio_init(VlAudio *audio);
void vl_audio_shutdown(VlAudio *audio);
int vl_audio_ready(const VlAudio *audio);
void vl_audio_update(VlAudio *audio);
void vl_audio_set_master_volume(float volume);

int vl_audio_bus_create(VlAudio *audio, const char *name, float volume);
void vl_audio_bus_set_volume(VlAudio *audio, int bus, float volume);
float vl_audio_bus_volume(const VlAudio *audio, int bus);
void vl_audio_bus_set_muted(VlAudio *audio, int bus, int muted);
int vl_audio_bus_muted(const VlAudio *audio, int bus);

VlAudioSound vl_audio_sound_load(VlAudio *audio, const char *path, int bus);
VlAudioSound vl_audio_sound_alias(VlAudio *audio, VlAudioSound source, int bus);
void vl_audio_sound_unload(VlAudioSound *sound);
void vl_audio_sound_play(const VlAudio *audio, VlAudioSound sound);
void vl_audio_sound_stop(VlAudioSound sound);
void vl_audio_sound_pause(VlAudioSound sound);
void vl_audio_sound_resume(VlAudioSound sound);
int vl_audio_sound_playing(VlAudioSound sound);
void vl_audio_sound_set_volume(VlAudioSound sound, float volume);
void vl_audio_sound_set_pitch(VlAudioSound sound, float pitch);
void vl_audio_sound_set_pan(VlAudioSound sound, float pan);

VlAudioMusic vl_audio_music_load(VlAudio *audio, const char *path, int bus);
void vl_audio_music_unload(VlAudioMusic *music);
void vl_audio_music_play(const VlAudio *audio, VlAudioMusic music);
void vl_audio_music_stop(VlAudioMusic music);
void vl_audio_music_pause(VlAudioMusic music);
void vl_audio_music_resume(VlAudioMusic music);
void vl_audio_music_seek(VlAudioMusic music, float seconds);
void vl_audio_music_set_volume(VlAudioMusic music, float volume);
void vl_audio_music_set_pitch(VlAudioMusic music, float pitch);
void vl_audio_music_set_pan(VlAudioMusic music, float pan);
void vl_audio_music_set_looping(VlAudioMusic music, int looping);
int vl_audio_music_playing(VlAudioMusic music);
float vl_audio_music_length(VlAudioMusic music);
float vl_audio_music_time(VlAudioMusic music);

VlAudioTone vl_audio_tone_create(float frequency, float seconds, float volume);
void vl_audio_tone_play(const VlAudio *audio, VlAudioTone tone, int bus);
void vl_audio_tone_unload(VlAudioTone *tone);
void vl_audio_play_spatial(const VlAudio *audio, VlAudioSound sound, int bus, float pan, float distance, float max_distance);

#endif
