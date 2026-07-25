// ============================================================================
//  Buffer B -- temporal accumulation (SVGF-style)
//  Reprojects last frame's accumulation using the camera the previous frame
//  actually used (Buffer A publishes both), variance-clips the history, and
//  accumulates the second luminance moment the wavelet passes need.
//
//  iChannel0: Buffer A (current noisy radiance, a = packed G-buffer)
//  iChannel1: Buffer B (history: rgb = colour, a = 2nd luminance moment)
// ============================================================================

vec4 fetchA(ivec2 q) {
    q = clamp(q, ivec2(0), ivec2(iResolution.xy) - 1);
    // (0,0) carries the camera state instead of radiance. Borrowing the pixel
    // next door is seamless and keeps this tap's packed G-buffer consistent with
    // the colour returned with it; substituting black instead leaves a visible
    // notch in the corner, where the wavelet kernels have fewer taps to dilute
    // it. The denoiser passes borrow the same way for the G-buffer -- both are
    // needed, and together they take that corner from 9/255 down to 3/255 on the
    // single pixel that is now showing its neighbour's value.
    if (q == ivec2(0)) q = ivec2(1, 0);
    return texelFetch(iChannel0, q, 0);
}
vec4 fetchB(ivec2 q) {
    q = clamp(q, ivec2(0), ivec2(iResolution.xy) - 1);
    return texelFetch(iChannel1, q, 0);
}
vec4 sampleB(vec2 fc) {   // manual bilinear (filter-setting independent)
    vec2 p = fc - 0.5;
    ivec2 i = ivec2(floor(p));
    vec2 f = p - vec2(i);
    return mix(mix(fetchB(i),               fetchB(i + ivec2(1, 0)), f.x),
               mix(fetchB(i + ivec2(0, 1)), fetchB(i + ivec2(1, 1)), f.x), f.y);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    ivec2 q = ivec2(fragCoord);
    vec4 cur = fetchA(q);

    // PSR depth from the packed G-buffer: no tracing in this pass
    float t; vec3 nP; int matP;
    unpackGbuf(cur.a, t, nP, matP);

    // ---- reproject into the previous frame ----------------------------------
    vec4 cam = camState(iChannel0);        // xy = this frame, zw = last frame
    vec3 ro, rd;
    camRay(fragCoord, iResolution, cam.xy, ro, rd);
    vec3 roP; mat3 cbP; float flP;
    getCam(cam.zw, roP, cbP, flP);

    vec3 dirP;
    // t is the unfolded distance to the PSR surface, so through the glass this
    // is a virtual point - which is the point: the history that belongs to a
    // refracted pixel moves with the refracted image, not with the ball.
    if (t > 0.0) dirP = (ro + rd * t) - roP;
    else         dirP = rd * 4000.0 + (ro - roP);  // degenerate tap: treat as far
    vec3 qc = dirP * cbP;                       // transpose(cb) * v
    vec2 fcPrev = vec2(-1.0);
    bool valid = qc.z > 1e-4 && iFrame > 0;
    if (valid) {
        vec2 uvn = qc.xy / qc.z * flP;
        fcPrev = uvn * iResolution.y + 0.5 * iResolution.xy;
        valid = fcPrev.x >= 0.5 && fcPrev.y >= 0.5 &&
                fcPrev.x <= iResolution.x - 0.5 && fcPrev.y <= iResolution.y - 0.5;
    }

    // ---- neighbourhood statistics for variance clipping ---------------------
    vec3 m1 = vec3(0.0), m2 = vec3(0.0);
    float lm1 = 0.0, lm2 = 0.0;
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        vec3 c = fetchA(q + ivec2(dx, dy)).rgb;
        m1 += c; m2 += c * c;
        float l = lum(c);
        lm1 += l; lm2 += l * l;
    }
    m1 /= 9.0; m2 /= 9.0; lm1 /= 9.0; lm2 /= 9.0;
    vec3 sig = sqrt(max(m2 - m1 * m1, 0.0));

    // firefly clip of the current sample against its neighbourhood; the
    // additive slack is tied to exposure so the threshold means the same thing
    // in display terms as it does in radiance
    float slack = clamp(0.8 / EXPOSURE, 0.02, 2.0);
    cur.rgb = min(cur.rgb, m1 + 3.5 * sig + slack);

    // Motion-adaptive accumulation: converge while the camera idles, clamp tight
    // and refresh fast while it is being dragged.
    //
    // The idle end is deliberately not as slow as it could be. A longer history
    // is quieter, but the orb pulses four times a second, and at alpha 0.07 the
    // filter's own lag swallowed four fifths of that pulse (measured: 12% of the
    // frame-mean swing reached the screen, against 60% in Buffer A). At 0.18 it
    // passes 20% for about 0.2/255 more frame-to-frame noise, which is the right
    // trade for a scene whose lighting actually moves.
    float motion = valid ? length(fcPrev - fragCoord) : 1e3;
    float mf = clamp((motion - 0.4) / 6.0, 0.0, 1.0);
    float GAMMA = mix(1.6, 1.0, mf);
    float alpha = mix(0.18, 0.38, mf);
    vec3 lo = m1 - GAMMA * sig, hi = m1 + GAMMA * sig;

    vec3 col; float mom2;
    if (valid) {
        vec4 hist = sampleB(fcPrev);
        vec3 hc = clamp(hist.rgb, lo, hi);
        col = mix(hc, cur.rgb, alpha);
        float l = lum(cur.rgb);
        mom2 = mix(max(hist.a, 0.0), l * l, alpha);
    } else {
        // bootstrap: publish the spatial variance so the wavelet passes filter
        // hard on the first frame instead of trusting a single sample
        col = cur.rgb;
        float l = lum(cur.rgb);
        float spatialVar = max(lm2 - lm1 * lm1, 0.0);
        mom2 = l * l + spatialVar * 2.0;
    }

    if (!(col.x >= 0.0) || !(col.y >= 0.0) || !(col.z >= 0.0)) { col = vec3(0.0); mom2 = 0.0; }
    fragColor = vec4(col, mom2);
}
