#include "audio.h"
#include <stdio.h>
int main(void){VlAudio a;if(!vl_audio_init(&a)){printf("audio device unavailable\n");return 0;}int sfx=VL_AUDIO_BUS_SFX;vl_audio_bus_set_volume(&a,sfx,.75f);VlAudioTone tone=vl_audio_tone_create(440.0f,.02f,.1f);printf("audio library initialized: buses=%d sfx=%.2f tone=%d\n",a.bus_count,vl_audio_bus_volume(&a,sfx),tone.valid);vl_audio_tone_unload(&tone);vl_audio_shutdown(&a);return 0;}
