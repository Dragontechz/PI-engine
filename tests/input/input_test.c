#include "input.h"
#include <stdio.h>
int main(void){VlInput input;vl_input_init(&input);printf("input library initialized: deadzone=%.2f rumble=%d battery_telemetry=%d\n",input.config.left_deadzone,vl_gamepad_has_rumble(),vl_gamepad_has_battery_telemetry());return input.config.left_deadzone>0.0f?0:1;}
