#include "audio.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static float clamp01_(float x) { return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }
static float bus_gain(const VlAudio *a, int id) {
    if (!a || id < 0 || id >= a->bus_count) return 1.0f;
    return a->buses[id].muted ? 0.0f : clamp01_(a->buses[id].volume);
}

int vl_audio_init(VlAudio *a) {
    if (!a) return 0;
    memset(a, 0, sizeof *a);
    InitAudioDevice();
    if (!IsAudioDeviceReady()) return 0;
    a->initialized = 1;
    a->bus_count = 1;
    a->buses[VL_AUDIO_BUS_MASTER] = (VlAudioBus){VL_AUDIO_BUS_MASTER, "master", 1.0f, 0};
    vl_audio_bus_create(a, "music", 1.0f);
    vl_audio_bus_create(a, "sfx", 1.0f);
    vl_audio_bus_create(a, "ui", 1.0f);
    vl_audio_bus_create(a, "ambience", 1.0f);
    return 1;
}
void vl_audio_shutdown(VlAudio *a) { if (a && a->initialized) { CloseAudioDevice(); a->initialized = 0; } }
int vl_audio_ready(const VlAudio *a) { return a && a->initialized && IsAudioDeviceReady(); }
void vl_audio_update(VlAudio *a) { (void)a; }
void vl_audio_set_master_volume(float v) { SetMasterVolume(clamp01_(v)); }
int vl_audio_bus_create(VlAudio *a, const char *name, float volume) { if (!a || a->bus_count >= VL_AUDIO_MAX_BUSES) return VL_AUDIO_INVALID_ID; int id=a->bus_count++; a->buses[id]=(VlAudioBus){id,"",clamp01_(volume),0}; if(name) snprintf(a->buses[id].name,sizeof a->buses[id].name,"%s",name); return id; }
void vl_audio_bus_set_volume(VlAudio *a,int id,float v){if(a&&id>=0&&id<a->bus_count)a->buses[id].volume=clamp01_(v);}
float vl_audio_bus_volume(const VlAudio *a,int id){return bus_gain(a,id);}
void vl_audio_bus_set_muted(VlAudio *a,int id,int m){if(a&&id>=0&&id<a->bus_count)a->buses[id].muted=m!=0;}
int vl_audio_bus_muted(const VlAudio *a,int id){return a&&id>=0&&id<a->bus_count&&a->buses[id].muted;}

VlAudioSound vl_audio_sound_load(VlAudio *a,const char *path,int bus){VlAudioSound s={0};s.bus=bus;s.base_volume=1.0f;if(vl_audio_ready(a)&&path){s.sound=LoadSound(path);s.valid=IsSoundValid(s.sound);}return s;}
VlAudioSound vl_audio_sound_alias(VlAudio *a,VlAudioSound src,int bus){VlAudioSound s={0};s.bus=bus;s.base_volume=src.base_volume;if(vl_audio_ready(a)&&src.valid){s.sound=LoadSoundAlias(src.sound);s.valid=IsSoundValid(s.sound);}return s;}
void vl_audio_sound_unload(VlAudioSound*s){if(s&&s->valid){UnloadSound(s->sound);s->valid=0;}}
void vl_audio_sound_play(const VlAudio*a,VlAudioSound s){if(a&&s.valid){SetSoundVolume(s.sound,s.base_volume*bus_gain(a,s.bus));PlaySound(s.sound);}}
void vl_audio_sound_stop(VlAudioSound s){if(s.valid)StopSound(s.sound);}
void vl_audio_sound_pause(VlAudioSound s){if(s.valid)PauseSound(s.sound);}
void vl_audio_sound_resume(VlAudioSound s){if(s.valid)ResumeSound(s.sound);}
int vl_audio_sound_playing(VlAudioSound s){return s.valid&&IsSoundPlaying(s.sound);}
void vl_audio_sound_set_volume(VlAudioSound s,float v){if(s.valid)SetSoundVolume(s.sound,clamp01_(v)*s.base_volume);}
void vl_audio_sound_set_pitch(VlAudioSound s,float v){if(s.valid)SetSoundPitch(s.sound,v>0?v:0.01f);}
void vl_audio_sound_set_pan(VlAudioSound s,float v){if(s.valid)SetSoundPan(s.sound,clamp01_(v));}

VlAudioMusic vl_audio_music_load(VlAudio*a,const char*p,int bus){VlAudioMusic m={0};m.bus=bus;m.base_volume=1.0f;if(vl_audio_ready(a)&&p){m.music=LoadMusicStream(p);m.valid=IsMusicValid(m.music);}return m;}
void vl_audio_music_unload(VlAudioMusic*m){if(m&&m->valid){UnloadMusicStream(m->music);m->valid=0;}}
void vl_audio_music_play(const VlAudio*a,VlAudioMusic m){if(a&&m.valid){SetMusicVolume(m.music,m.base_volume*bus_gain(a,m.bus));PlayMusicStream(m.music);}}
void vl_audio_music_stop(VlAudioMusic m){if(m.valid)StopMusicStream(m.music);}
void vl_audio_music_pause(VlAudioMusic m){if(m.valid)PauseMusicStream(m.music);}
void vl_audio_music_resume(VlAudioMusic m){if(m.valid)ResumeMusicStream(m.music);}
void vl_audio_music_seek(VlAudioMusic m,float s){if(m.valid)SeekMusicStream(m.music,s);}
void vl_audio_music_set_volume(VlAudioMusic m,float v){if(m.valid)SetMusicVolume(m.music,clamp01_(v)*m.base_volume);}
void vl_audio_music_set_pitch(VlAudioMusic m,float v){if(m.valid)SetMusicPitch(m.music,v>0?v:0.01f);}
void vl_audio_music_set_pan(VlAudioMusic m,float v){if(m.valid)SetMusicPan(m.music,clamp01_(v));}
void vl_audio_music_set_looping(VlAudioMusic m,int loop){if(m.valid)m.music.looping=loop!=0;}
int vl_audio_music_playing(VlAudioMusic m){return m.valid&&IsMusicStreamPlaying(m.music);}
float vl_audio_music_length(VlAudioMusic m){return m.valid?GetMusicTimeLength(m.music):0.0f;}
float vl_audio_music_time(VlAudioMusic m){return m.valid?GetMusicTimePlayed(m.music):0.0f;}

VlAudioTone vl_audio_tone_create(float frequency,float seconds,float volume){VlAudioTone t={0};if(frequency<=0||seconds<=0)return t;int rate=44100,count=(int)(seconds*rate);if(count<1)count=1;float*data=(float*)MemAlloc((unsigned int)count*sizeof(float));if(!data)return t;for(int i=0;i<count;i++){float env=1.0f-(float)i/(float)count;data[i]=sinf(2.0f*PI*frequency*(float)i/(float)rate)*volume*env;}t.wave=(Wave){(unsigned int)count,rate,32,1,data};t.sound=LoadSoundFromWave(t.wave);t.valid=IsSoundValid(t.sound);return t;}
void vl_audio_tone_play(const VlAudio*a,VlAudioTone t,int bus){if(a&&t.valid){SetSoundVolume(t.sound,bus_gain(a,bus));PlaySound(t.sound);}}
void vl_audio_tone_unload(VlAudioTone*t){if(t&&t->valid){UnloadSound(t->sound);UnloadWave(t->wave);t->valid=0;}}
void vl_audio_play_spatial(const VlAudio*a,VlAudioSound s,int bus,float pan,float distance,float max_distance){if(!a||!s.valid)return;float attenuation=max_distance>0?1.0f-clamp01_(distance/max_distance):1.0f;SetSoundPan(s.sound,clamp01_(pan));SetSoundVolume(s.sound,attenuation*bus_gain(a,bus));PlaySound(s.sound);}
