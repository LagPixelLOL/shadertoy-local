// Texture channels. This project binds three different builtin textures, so
// it needs no asset files and renders identically on any machine.
//
//   iChannel0 = checker      (nearest, repeat)  - shows wrapping and filtering
//   iChannel1 = uv           (linear,  clamp )  - red = u, green = v
//   iChannel2 = noise        (linear,  repeat)  - deterministic RGBA noise
//
// iChannelResolution[N] holds each channel's pixel size.
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    // Three vertical strips, one per channel.
    float strip = floor(uv.x * 3.0);
    vec2 local = vec2(fract(uv.x * 3.0), uv.y);

    // Scroll and zoom so wrapping behaviour is visible.
    vec2 st = local * 2.0 + vec2(iTime * 0.1, 0.0);

    vec3 col;
    if (strip < 0.5) {
        col = texture(iChannel0, st).rgb;
    } else if (strip < 1.5) {
        // Clamped: sampling beyond [0,1] repeats the edge pixel.
        col = texture(iChannel1, st).rgb;
    } else {
        col = texture(iChannel2, st).rgb;
    }

    // Thin dividers between strips.
    float edge = min(local.x, 1.0 - local.x);
    col *= smoothstep(0.0, 0.008, edge);

    fragColor = vec4(col, 1.0);
}
