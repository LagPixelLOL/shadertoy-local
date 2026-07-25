// Classic animated plasma. Single pass, no inputs -- the simplest possible
// project shape (image.glsl alone, no config file needed).
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Aspect-corrected coordinates centred on the screen.
    vec2 p = (2.0 * fragCoord - iResolution.xy) / iResolution.y;

    float v = 0.0;
    v += sin((p.x + iTime) * 3.0);
    v += sin((p.y + iTime * 0.7) * 4.0);
    v += sin((p.x + p.y + iTime * 0.5) * 5.0);
    v += sin(length(p) * 8.0 - iTime * 2.0);
    v *= 0.25;

    vec3 col = 0.5 + 0.5 * cos(6.2831 * (v + vec3(0.0, 0.33, 0.67)));

    fragColor = vec4(col, 1.0);
}
