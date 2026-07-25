// ============================================================================
//  Buffer C -- denoiser: edge-aware a-trous wavelet pass 1 (step = 1)
//  Turns the accumulated moments into a variance estimate; the depth, normal
//  and material of every tap come from Buffer A's packed G-buffer, and the tap's
//  world position is rebuilt from the deterministic camera ray.
//
//  iChannel0: Buffer B (rgb = accumulated colour, a = 2nd luminance moment)
//  iChannel1: Buffer A (alpha = packed G-buffer, (0,0) = camera state)
//  output: rgb = filtered colour, a = filtered variance
// ============================================================================

#define STEP 1

vec4 fetchC0(ivec2 q) {
    q = clamp(q, ivec2(0), ivec2(iResolution.xy) - 1);
    return texelFetch(iChannel0, q, 0);
}
// packed PSR G-buffer from Buffer A. (0,0) holds the camera state there, so its
// alpha is an orbit angle rather than a packed G-buffer; borrow the pixel next
// door, exactly as Buffer B does for the radiance.
void gbufAt(ivec2 q, out float t, out vec3 n, out int mat) {
    q = clamp(q, ivec2(0), ivec2(iResolution.xy) - 1);
    if (q == ivec2(0)) q = ivec2(1, 0);
    unpackGbuf(texelFetch(iChannel1, q, 0).a, t, n, mat);
}
float varianceAt(ivec2 q) {
    vec4 c = fetchC0(q);
    return max(c.a - lum(c.rgb) * lum(c.rgb), 0.0);
}
// 3x3 prefiltered variance (SVGF)
float filteredVariance(ivec2 q) {
    float k[3]; k[0] = 0.25; k[1] = 0.125; k[2] = 0.0625;
    float s = 0.0, wsum = 0.0;
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        float w = k[abs(dx) + abs(dy)];
        s += w * varianceAt(q + ivec2(dx, dy));
        wsum += w;
    }
    return s / wsum;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    ivec2 q = ivec2(fragCoord);
    vec4 cC = fetchC0(q);
    float vC = filteredVariance(q);
    float lC = lum(cC.rgb);

    float tC; vec3 nC; int mC;
    gbufAt(q, tC, nC, mC);
    vec2 ang = camState(iChannel1).xy;       // orbit angles (Buffer A)
    vec3 roC, rdC; camRay(fragCoord, iResolution, ang, roC, rdC);
    vec3 pC = roC + rdC * max(tC, 0.0);

    // the luminance tolerance follows the estimated noise, so converged regions
    // stop being blurred while noisy ones still are
    float phiL = 4.0 * sqrt(vC) + 1e-3;
    float phiZ = 0.06 * float(STEP) * (1.0 + 0.06 * max(tC, 0.0));

    float kern[3]; kern[0] = 0.375; kern[1] = 0.25; kern[2] = 0.0625;
    vec3 sumC = vec3(0.0);
    float sumV = 0.0, sumW = 0.0;
    for (int dy = -2; dy <= 2; dy++)
    for (int dx = -2; dx <= 2; dx++) {
        ivec2 qt = q + ivec2(dx, dy) * STEP;
        vec4 cT = fetchC0(qt);
        float w = kern[abs(dx)] * kern[abs(dy)];
        if (dx != 0 || dy != 0) {
            vec2 fcT = vec2(qt) + 0.5;
            float tT; vec3 nT; int mT;
            gbufAt(qt, tT, nT, mT);
            if (mT != mC) w *= 0.05;
            if (tC > 0.0 && tT > 0.0) {
                vec3 roT, rdT; camRay(fcT, iResolution, ang, roT, rdT);
                vec3 pT = roT + rdT * tT;
                float dz = abs(dot(pT - pC, nC));
                w *= exp(-dz / phiZ);
                w *= pow(max(dot(nC, nT), 0.0), 32.0);
            }
            float lT = lum(cT.rgb);
            w *= exp(-abs(lT - lC) / phiL);
        }
        sumC += w * cT.rgb;
        sumV += w * w * varianceAt(qt);
        sumW += w;
    }
    fragColor = vec4(sumC / sumW, sumV / (sumW * sumW));
}
