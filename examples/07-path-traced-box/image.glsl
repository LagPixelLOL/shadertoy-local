// ============================================================================
//  Image -- denoiser: final a-trous pass (step = 4), then put the albedo back,
//  expose, tonemap and dither.
//
//  iChannel0: Buffer D (rgb = filtered colour, a = variance)
//  iChannel1: Buffer A (alpha = packed G-buffer, (0,0) = camera state)
//
//  Buffers mean this shader has to be run forward from frame 0, which the
//  renderer does by default (--precharge all):
//      shadertoy render -C examples/07-path-traced-box --frame 240
//  Drag the mouse to aim the camera:
//      shadertoy render -C examples/07-path-traced-box --frame 240 \
//          --input '[{"frame":0,"op":"mouse_down","pos":[120,300]}]'
// ============================================================================

#define STEP 4

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

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    ivec2 q = ivec2(fragCoord);
    vec4 cC = fetchC0(q);
    float vC = max(cC.a, 0.0);
    float lC = lum(cC.rgb);

    float tC; vec3 nC; int mC;
    gbufAt(q, tC, nC, mC);
    vec2 ang = camState(iChannel1).xy;       // orbit angles (Buffer A)
    vec3 roC, rdC; camRay(fragCoord, iResolution, ang, roC, rdC);
    vec3 pC = roC + rdC * max(tC, 0.0);

    float phiL = 4.0 * sqrt(vC) + 1e-3;
    float phiZ = 0.06 * float(STEP) * (1.0 + 0.06 * max(tC, 0.0));

    float kern[3]; kern[0] = 0.375; kern[1] = 0.25; kern[2] = 0.0625;
    vec3 sumC = vec3(0.0);
    float sumW = 0.0;
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
        sumW += w;
    }
    vec3 col = sumC / sumW;

    // ---- coverage-aware supersampled remodulation -----------------------------
    // The filtered signal is irradiance, one value per pixel. Multiplying it by
    // one albedo lookup would leave every material edge as hard as the pixel
    // grid. Instead 4 rotated-grid subsamples each trace the PSR surface they
    // land on: same surface as the pixel centre -> gradient-reconstructed
    // irradiance; different surface (a silhouette, a ball's rim) -> irradiance
    // is pulled from the neighbouring pixel that does lie on that surface.
    vec3 gx = dFdx(col), gy = dFdy(col);
    vec2 offs[4];
    offs[0] = vec2(-0.375, -0.125); offs[1] = vec2( 0.125, -0.375);
    offs[2] = vec2( 0.375,  0.125); offs[3] = vec2(-0.125,  0.375);
    float pixAng = pixelAngle(iResolution);
    vec3 acc = vec3(0.0);
    for (int i = 0; i < 4; i++) {
        float tS; vec3 nS; int mS; vec3 posS, dirS;
        primaryHitFull(fragCoord + offs[i], iResolution, ang, tS, nS, mS, posS, dirS);
        float fw = tS * pixAng / max(abs(dot(nS, dirS)), 0.45);
        vec3 irr = max(col + gx * offs[i].x + gy * offs[i].y, 0.0);
        bool other = (mS != mC) ||
                     (tS > 0.0 && tC > 0.0 && abs(tS - tC) > 0.08 * max(tS, tC));
        if (other) {
            ivec2 d1 = ivec2(offs[i].x > 0.0 ? 1 : -1, 0);
            ivec2 d2 = ivec2(0, offs[i].y > 0.0 ? 1 : -1);
            float tN; vec3 nN; int mN;
            gbufAt(q + d1, tN, nN, mN);
            if (mN == mS) irr = fetchC0(q + d1).rgb;
            else {
                gbufAt(q + d2, tN, nN, mN);
                if (mN == mS) irr = fetchC0(q + d2).rgb;
                else {
                    gbufAt(q + d1 + d2, tN, nN, mN);
                    if (mN == mS) irr = fetchC0(q + d1 + d2).rgb;
                }
            }
        }
        acc += irr * psrAlbedo(mS, posS, fw);
    }
    col = acc * 0.25;

    // ---- exposure / tonemap / display ----------------------------------------
    col *= EXPOSURE;
    col = ACES(col);
    // subtle vignette
    vec2 uv = fragCoord / iResolution.xy;
    col *= 1.0 - 0.18 * pow(length(uv - 0.5) * 1.35, 3.0);
    col = pow(col, vec3(1.0 / 2.2));
    // dither: hides 8-bit banding across the smooth wall gradients
    col += (hash12(fragCoord + fract(iTime) * 113.1) - 0.5) * (1.2 / 255.0);

    fragColor = vec4(col, 1.0);
}
