// Image pass. Uniforms are read HERE, where they legally exist, and handed to
// the Common helpers as a struct.
//
// Compare with any other example: those reference iTime directly inside
// common.glsl, which works locally and on the site, but makes the site's
// Common tab light up red. Run `shadertoy check` on both to see the difference.
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // The single point where pass uniforms are captured.
    ST u = ST_CAPTURE;

    vec2 p = centred(u, fragCoord);
    vec2 m = (cursor(u) * 2.0 - 1.0) * vec2(aspect(u), 1.0);

    // A ring that follows the cursor, plus an animated background.
    float d = sdCircle(p - m, 0.35 + 0.05 * sin(u.time * 2.0));
    float ring = smoothstep(0.02, 0.0, abs(d));

    vec3 col = palette(0.1 * u.time + 0.25 * length(p));
    col *= 0.35 + 0.65 * smoothstep(0.6, 0.0, length(p - m));
    col = mix(col, vec3(1.0), ring);

    fragColor = vec4(col, 1.0);
}
