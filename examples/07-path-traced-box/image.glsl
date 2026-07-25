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
// packed PSR G-buffer from Buffer A. (0,0) holds the camera state there, so its
// alpha is an orbit angle rather than a packed G-buffer; borrow the pixel next
// door, exactly as Buffer B does for the radiance.
void gbufAt(ivec2 q, out float t, out vec3 n, out int mat) {
    q = clamp(q, ivec2(0), ivec2(iResolution.xy) - 1);
    if (q == ivec2(0)) q = ivec2(1, 0);
    unpackGbuf(texelFetch(iChannel1, q, 0).a, t, n, mat);
}

bool surfaceMatches(ivec2 q, int mat, vec3 n) {
    float tN; vec3 nN; int mN;
    gbufAt(q, tN, nN, mN);
    return mN == mat && dot(nN, n) > NORMAL_MATCH;
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
    // grid. Instead each subsample traces the PSR surface it lands on: same
    // surface as the pixel centre -> gradient-reconstructed irradiance; different
    // surface (a silhouette, a ball's rim) -> irradiance is pulled from the
    // neighbouring pixel that does lie on that surface. The albedo then varies
    // between subsamples, which is what anti-aliases the silhouette.
    //
    // Coverage quantises to SUBS+1 levels, so the sample count sets how smooth an
    // edge can be. Measured against a 4x4 supersampled reference, on the pixels
    // the gate flags, and as frame-to-frame change on those same pixels while the
    // camera orbits (edge crawl, which a single frame cannot show):
    //
    //     samples    error    crawl
    //        1       10.85     4.76
    //        4        8.56     4.16
    //        9        8.40     3.62
    //       16        8.33     3.42
    //
    // Error plateaus after 4 -- past that the residual is not coverage but the
    // borrowed neighbour irradiance below -- while crawl keeps falling, and this
    // camera is always moving. 16 costs nothing measurable over 4 (both sit in the
    // noise around 0.45 ms against a 0.32 ms baseline) because the gate below
    // keeps supersampling off the ~96% of pixels that do not need it: the pixel is
    // first tested against its 4-neighbourhood in the G-buffer, one texelFetch
    // each, and only pixels straddling a discontinuity supersample at all.
    // Interior pixels take a single lookup, which is all they need -- the checker
    // inside a face is already filtered analytically over the footprint.
    vec3 gx = dFdx(col), gy = dFdy(col);
    float pixAng = pixelAngle(iResolution);

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
        primaryHitFull(fragCoord + off, iResolution, ang, tS, nS, mS, posS, dirS);
        float fw = tS * pixAng / max(abs(dot(nS, dirS)), 0.45);
        vec3 irr = max(col + gx * off.x + gy * off.y, 0.0);
        bool other = (mS != mC) || (dot(nS, nC) < NORMAL_MATCH) ||
                     (tS > 0.0 && tC > 0.0 && abs(tS - tC) > 0.08 * max(tS, tC));
        if (other) {
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
        acc += irr * psrAlbedo(mS, posS, fw);
    }
    col = acc / float(subs);

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
