// Progressive accumulation: every frame since the last orientation change has
// equal weight. No sliding history, firefly clipping, or neighborhood clamping.
// RGB slots hold mean CIE XYZ; alpha is the mean squared per-frame Y luminance.
// iChannel0: Buffer A. iChannel1: previous Buffer B.

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    ivec2 q = ivec2(fragCoord);
    if (q.y == 0 && q.x < 2) q = ivec2(2, 0);
    vec3 current = texelFetch(iChannel0, q, 0).rgb;
    float l = lum(current);
    vec4 sampleValue = vec4(current, l * l);
    float count = accumulationCount(crystalState(iChannel0), iFrame);
    if (count <= 1.0) {
        fragColor = sampleValue;
    } else {
        vec4 history = texelFetch(iChannel1, q, 0);
        fragColor = history + (sampleValue - history) / count;
    }
}
