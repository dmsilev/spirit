vec3 colormap(vec3 direction) {
     vec3 color_down = vec3(0.0, 0.0, 1.0);
     vec3 color_up = vec3(1.0, 0.0, 0.0);
     return mix(color_down, color_up, direction.z*0.5+0.5);
}
