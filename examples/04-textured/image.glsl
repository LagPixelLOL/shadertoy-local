// Channel inputs, and which of them exist on shadertoy.com.
//
// Four strips, left to right:
//
//   iChannel0  textures/pattern.png   a real file -- the PORTABLE way to supply
//                                     a texture. On shadertoy.com you would
//                                     upload it or pick a stock texture.
//   iChannel1  rgba-noise-medium      mirrors the site's "RGBA Noise Medium"
//                                     (256x256). Same role and size; the actual
//                                     pixel values differ, because Shadertoy's
//                                     assets cannot be redistributed.
//   iChannel2  bayer                  a 16x16 ordered-dither matrix. This one is
//                                     EXACT: a Bayer matrix is defined by
//                                     recurrence, not authored, so it is
//                                     identical to the site's.
//   iChannel3  uv                     a shadertoy-local debug aid with NO
//                                     equivalent on the site. Red = u, green = v,
//                                     which makes orientation and wrap mode
//                                     obvious at a glance.
//
// iChannelResolution[N] holds each channel's pixel size.
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    float strip = floor(uv.x * 4.0);
    vec2 local = vec2(fract(uv.x * 4.0), uv.y);

    // Scroll and zoom so wrapping behaviour is visible.
    vec2 st = local * 2.0 + vec2(iTime * 0.1, 0.0);

    vec3 col;
    if (strip < 0.5) {
        col = texture(iChannel0, st).rgb;
    } else if (strip < 1.5) {
        col = texture(iChannel1, st).rgb;
    } else if (strip < 2.5) {
        // Ordered dithering, the reason a Bayer texture exists. Quantise a
        // smooth ramp to 3 levels using the matrix as a threshold.
        float ramp = local.y;
        float threshold = texelFetch(iChannel2, ivec2(fragCoord) % 16, 0).r;
        col = vec3(floor(ramp * 3.0 + threshold) / 3.0);
    } else {
        col = texture(iChannel3, st).rgb;
    }

    // Thin dividers between strips.
    float edge = min(local.x, 1.0 - local.x);
    col *= smoothstep(0.0, 0.008, edge);

    fragColor = vec4(col, 1.0);
}
