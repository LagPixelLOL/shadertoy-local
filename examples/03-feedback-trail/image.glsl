// Image pass: tone-maps and vignettes Buffer A.
//
// iChannel0 is wired to buffer_a. Because the image pass runs AFTER the
// buffer passes, this reads Buffer A's output from the CURRENT frame -- unlike
// buffer_a.glsl, which reads its own previous frame.
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    vec3 col = texture(iChannel0, uv).rgb;

    // The accumulator is unbounded, so tone-map instead of clipping.
    col = col / (1.0 + col);

    // Vignette.
    vec2 q = uv - 0.5;
    col *= 1.0 - 0.7 * dot(q, q);

    col = pow(col, vec3(0.4545));   // linear -> sRGB
    fragColor = vec4(col, 1.0);
}
