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

    // Motion-adaptive accumulation: converge hard while the camera idles, clamp
    // tight and refresh fast while it is being dragged.
    //
    // The idle end can afford to be this slow because nothing in this scene
    // changes fast. The lighting is static apart from a hue rotation that takes
    // 15 seconds, so a history worth ~14 frames costs nothing in lag and buys
    // the quietest image; the only thing the filter has to keep up with is the
    // camera, which the motion term already handles.
    // GAMMA bounds the history to the current frame's neighbourhood: it is what
    // stops stale history ghosting when the camera moves. It is loose only while
    // nothing moves at all, and clamps down to 1 as soon as anything does.
    //
    // The ramp is deliberately abrupt -- fully tight by half a pixel of
    // reprojected motion -- and that number was expensive to learn. A lazier ramp
    // (tight only past 4 px) leaves GAMMA near 7 during a slow drag, which is
    // exactly where stale history survives: measured against a reference with the
    // history disabled, a slow drag went from 38.9 to 45.3 at the 99th percentile,
    // and that is visible as smearing. A fast drag hides the problem, because both
    // ends of the ramp are tight by then, so it has to be tested slow.
    //
    // Judge this only against a reference that has no history AND no clamps of its
    // own. A high-spp render through this same pipeline is not ground truth: at
    // high sample counts sig shrinks, so its own clamp bites harder than the one
    // being measured, and comparing against it will appear to justify loosening
    // GAMMA far past the point where it starts ghosting.
    //
    // Both thresholds are in 360p-equivalent pixels: the same camera motion
    // displaces four times as many pixels at 1440p, and neither the ghosting nor
    // the convergence behaviour should change with the output size.
    float motion = valid ? length(fcPrev - fragCoord) : 1e3;
    float mpx = motion * (360.0 / iResolution.y);
    float mfA = clamp((mpx - 0.4) / 6.0, 0.0, 1.0);   // alpha: how fast to refresh
    float GAMMA = mix(10.0, 1.0, clamp(mpx / 0.5, 0.0, 1.0));
    float alpha = mix(0.07, 0.38, mfA);
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
