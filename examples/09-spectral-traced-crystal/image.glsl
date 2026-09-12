// ============================================================================
//  Image -- denoiser: final a-trous pass (step = 4), then put the albedo back,
//  convert CIE XYZ to display RGB, expose, tonemap and dither.
//
//  iChannel0: Buffer D (rgb = filtered XYZ, a = variance)
//  iChannel1: Buffer A (alpha = packed G-buffer, first two pixels = crystal state)
//
//  Buffers mean this shader has to be run forward from frame 0, which the
//  renderer does by default (--precharge all):
//      shadertoy render -C examples/09-spectral-traced-crystal --frame 240 --output /tmp/opencode/crystal
//  Drag for pitch/yaw; Shift + horizontal drag for roll. The camera never moves.
//  Accumulation continues without a history cap until the crystal rotates:
//      shadertoy render -C examples/09-spectral-traced-crystal --frame 240 --output /tmp/opencode/crystal \
//          --input '[{"frame":0,"op":"mouse_down","pos":[320,180]}, \
//                    {"frame":1,"op":"mouse_move","pos":[370,180]}]'
// ============================================================================

#define STEP 4

// Subsamples used for coverage at material edges: a k*k grid rotated off the
// axes, which is where quantised coverage shows worst. SUBS must stay k*k.
#define SUB_K 4
#define SUBS  (SUB_K * SUB_K)

// Two surfaces count as the same one when material and normal both agree. The
// packed normal is 6 bits per octahedral axis, good to a couple of degrees, so
// this threshold is far above the quantisation.
#define NORMAL_MATCH 0.9

vec2 subOffset(int i) {
    vec2 g = (vec2(i % SUB_K, i / SUB_K) + 0.5) / float(SUB_K) - 0.5;
    const float CA = 0.894427, SA = 0.447214;   // atan(1/2), the usual RGSS angle
    return vec2(g.x * CA - g.y * SA, g.x * SA + g.y * CA);
}

vec4 fetchC0(ivec2 q) {
    q = clamp(q, ivec2(0), ivec2(iResolution.xy) - 1);
    return texelFetch(iChannel0, q, 0);
}
// Skip the two state pixels, just as Buffer B does for radiance.
void gbufAt(ivec2 q, out float t, out vec3 n, out int mat) {
    q = clamp(q, ivec2(0), ivec2(iResolution.xy) - 1);
    if (q.y == 0 && q.x < 2) q = ivec2(2, 0);
    unpackGbuf(texelFetch(iChannel1, q, 0).a, t, n, mat);
}

bool surfaceMatches(ivec2 q, int mat, vec3 n) {
    float tN; vec3 nN; int mN;
    gbufAt(q, tN, nN, mN);
    return mN == mat && dot(nN, n) > NORMAL_MATCH;
}


void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    setCrystalRotation(crystalState(iChannel1).rotation);
    ivec2 q = ivec2(fragCoord);
    vec4 cC = fetchC0(q);
    float vC = max(cC.a, 0.0);
    float lC = lum(cC.rgb);

    float tC; vec3 nC; int mC;
    gbufAt(q, tC, nC, mC);
    vec3 roC, rdC; camRay(fragCoord, iResolution, roC, rdC);
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
                vec3 roT, rdT; camRay(fcT, iResolution, roT, rdT);
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
    // grid. Instead each subsample traces the PSR surface it lands on: same
    // surface as the pixel centre -> gradient-reconstructed irradiance; different
    // surface (a silhouette or a crystal edge) -> irradiance is pulled from the
    // neighbouring pixel that does lie on that surface. The albedo then varies
    // between subsamples, which is what anti-aliases the silhouette.
    //
    // Only pixels straddling a G-buffer discontinuity need supersampling.
    // Interior pixels take one lookup; each wall has a constant white material.
    vec3 gx = dFdx(col), gy = dFdy(col);

    // Reuse Buffer A's centre-surface albedo so demodulation cancels exactly.
    // Subsamples need their own albedo only when they land on another surface.
    float tK; vec3 nK; int mK; vec3 posK, dirK;
    primaryHitFull(fragCoord, iResolution, tK, nK, mK, posK, dirK);
    vec2 fwK = psrFootprint(fragCoord, iResolution, tK, nK, mK, posK, dirK);
    float albC = psrAlbedo(mK, posK, fwK);

    ivec2 nb[4];
    nb[0] = ivec2(1, 0); nb[1] = ivec2(-1, 0);
    nb[2] = ivec2(0, 1); nb[3] = ivec2(0, -1);
    bool edge = false;
    for (int k = 0; k < 4; k++) {
        float tN; vec3 nN; int mN;
        gbufAt(q + nb[k], tN, nN, mN);
        if (mN != mC) edge = true;
        if (tN > 0.0 && tC > 0.0 && abs(tN - tC) > 0.02 * max(tN, tC)) edge = true;
        // Normals too, not just material and depth. The crease between two faces
        // of the same block is continuous in both of those, so a material-only
        // test walks straight past the worst aliasing in the frame: the two faces
        // carry different irradiance, and without this the pixel that straddles
        // them just picks one.
        if (dot(nN, nC) < NORMAL_MATCH) edge = true;
    }
    int subs = edge ? SUBS : 1;

    vec3 acc = vec3(0.0);
    for (int i = 0; i < SUBS; i++) {
        if (i >= subs) break;
        vec2 off = edge ? subOffset(i) : vec2(0.0);
        float tS; vec3 nS; int mS; vec3 posS, dirS;
        primaryHitFull(fragCoord + off, iResolution, tS, nS, mS, posS, dirS);
        vec3 irr = max(col + gx * off.x + gy * off.y, 0.0);
        float alb = albC;
        bool other = (mS != mC) || (dot(nS, nC) < NORMAL_MATCH) ||
                     (tS > 0.0 && tC > 0.0 && abs(tS - tC) > 0.08 * max(tS, tC));
        if (other) {
            // a different surface: it needs its own albedo, at its own footprint
            vec2 fw = psrFootprint(fragCoord + off, iResolution,
                                   tS, nS, mS, posS, dirS);
            alb = psrAlbedo(mS, posS, fw);
            // This subsample sees a different surface than the pixel centre, so
            // the centre's irradiance is the wrong number for it. Borrow from a
            // neighbour that does lie on that surface, trying the two axes and
            // then the diagonal; matching on normal as well as material is what
            // lets the two sides of a crease find their own neighbour.
            ivec2 d1 = ivec2(off.x > 0.0 ? 1 : -1, 0);
            ivec2 d2 = ivec2(0, off.y > 0.0 ? 1 : -1);
            if (surfaceMatches(q + d1, mS, nS))      irr = fetchC0(q + d1).rgb;
            else if (surfaceMatches(q + d2, mS, nS)) irr = fetchC0(q + d2).rgb;
            else if (surfaceMatches(q + d1 + d2, mS, nS))
                                                     irr = fetchC0(q + d1 + d2).rgb;
        }
        acc += irr * alb;
    }
    col = acc / float(subs);

    // ---- exposure / tonemap / display ----------------------------------------
    // Compress out-of-gamut spectra toward neutral at constant luminance, rather
    // than clipping channels. One shared tone scale preserves the remaining RGB
    // ratios; independent curves flatten bright rainbows into cyan/yellow/white.
    vec3 rgb = xyzToLinearSRGB(col);
    float Y = max(col.y, 0.0);
    float lowest = min(min(rgb.r, rgb.g), min(rgb.b, 0.0));
    rgb = max(vec3(Y) + (rgb - vec3(Y)) * (Y / max(Y - lowest, 1e-8)), 0.0);
    float peak = max(max(rgb.r, rgb.g), rgb.b);
    col = rgb * (ACES(vec3(peak * EXPOSURE)).x / max(peak, 1e-8));
    // subtle vignette
    vec2 uv = fragCoord / iResolution.xy;
    col *= 1.0 - 0.18 * pow(length(uv - 0.5) * 1.35, 3.0);
    col = mix(12.92 * col, 1.055 * pow(col, vec3(1.0 / 2.4)) - 0.055,
              step(vec3(0.0031308), col));
    // Static dither hides banding without adding temporal noise after convergence.
    col += (hash12(fragCoord) - 0.5) * (1.2 / 255.0);

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
