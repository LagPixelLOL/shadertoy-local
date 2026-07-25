// ============================================================================
//  Buffer A -- path tracer: 2 spp/frame, next event estimation against both
//  emitters, delta chains through the mirror and glass balls (which is where the
//  caustics come from), Beer-Lambert absorption inside the glass, and a
//  volumetric walk through the frosted globe.
//
//  iChannel0: Buffer A (self -- camera state pixel at (0,0))
//  output: rgb = radiance / albedo (linear HDR), a = packed G-buffer
//  pixel (0,0): xy = this frame's orbit angles, zw = the previous frame's
// ============================================================================

#define MAX_BOUNCE 8
#define SPP 2

// Firefly control, split by path type -- and it has to be split.
//
// The filament is (R/r)^2 = 17x brighter than the globe's envelope, which is
// forced by energy conservation, not chosen. A path that bounces off the floor,
// refracts through the glass ball and then happens to punch through the medium
// without scattering (probability e^-sigma_t*d, a few percent) comes back with
// that radiance on it. The expectation is right and the variance is appalling:
// isolated white dots all over the floor, which the wavelet passes then smear
// into blotches.
//
// So indirect radiance -- anything gathered after the first non-delta bounce --
// is clamped hard, while radiance still on the delta chain is left alone. That
// keeps what the camera is actually looking at (the globe's hot centre, its
// mirror image, its image through the glass) at full strength, and only bounds
// the paths that cannot be resolved at 2 spp. Clamping indirect light biases
// caustics slightly dark; that is the accepted trade, and it is what production
// renderers do for the same reason.
#define CLAMP_INDIRECT 6.0
#define FIREFLY_MAX 400.0

// --- direct light. Uniform area sampling of the panel, converted from area to
//     solid angle by the cos'/r^2 geometry term. The panel is small, flat and
//     untextured, so there is nothing a cleverer measure would buy.
vec3 directLight(vec3 p, vec3 n, vec3 v, vec3 alb, float rough) {
    vec3 lp = lightPoint(rnd2());
    vec3 d = lp - p;
    float r2 = dot(d, d);
    if (r2 < 1e-6) return vec3(0.0);
    float r = sqrt(r2);
    d /= r;
    float NoL = dot(n, d);
    float cosL = d.y;                    // the panel faces straight down
    if (NoL <= 0.0 || cosL <= 1e-4) return vec3(0.0);
    // tmax stops just short of the panel so the panel is not its own occluder;
    // the glass ball IS an occluder here, and its hard shadow is correct - the
    // light that really gets through it arrives as a caustic, below.
    if (sceneOccluded(p + n * 1e-4, d, r - 1e-3)) return vec3(0.0);
    return evalBRDF(n, v, d, alb, rough) * NoL * LIGHT_E * (cosL * LIGHT_A / r2);
}

// --- the frosted globe, sampled over the cone it subtends from the shading
//     point. Uniform in solid angle inside that cone is the textbook estimator
//     for a spherical emitter and strictly better than sampling its area: not
//     one sample is wasted on the hemisphere facing away. It also keeps the
//     contact region well behaved -- as a shading point approaches the surface
//     the cone opens to a hemisphere, instead of a point light's 1/r^2 blowing
//     up. The globe is treated here as a uniform emitter; see common.glsl for
//     why that is energy-exact against the medium walk.
vec3 globeLight(vec3 p, vec3 n, vec3 v, vec3 alb, float rough, float time) {
    vec3 oc = GLOBE_C - p;
    float d2 = dot(oc, oc);
    float r2 = GLOBE_R * GLOBE_R;
    if (d2 <= r2 * 1.0004) return vec3(0.0);   // shading point on/inside it
    float cosMax = sqrt(max(1.0 - r2 / d2, 0.0));
    vec3 l = sampleCone(oc * inversesqrt(d2), cosMax, rnd2());
    float NoL = dot(n, l);
    if (NoL <= 0.0) return vec3(0.0);
    // distance to the globe along l; the sample is inside the cone, so it hits
    float b = -dot(oc, l);
    float disc = b * b - (d2 - r2);
    if (disc <= 0.0) return vec3(0.0);         // grazing miss from rounding
    float tHit = -b - sqrt(disc);
    if (tHit <= 1e-4) return vec3(0.0);
    if (sceneOccluded(p + n * 1e-4, l, tHit - 1e-3)) return vec3(0.0);
    float solid = TAU * (1.0 - cosMax);        // 1 / pdf
    return evalBRDF(n, v, l, alb, rough) * NoL * globeRadiance(time) * solid;
}

// --- volumetric walk through the globe's interior: free-flight distance
//     sampling in a homogeneous, purely scattering medium, isotropic phase
//     function, with next event estimation against the filament at every scatter
//     event. Because the medium does not absorb, the single-scatter albedo is 1
//     and the sampling weight cancels exactly -- there is no throughput term to
//     carry until russian roulette starts.
//
//     Two things are deliberately left out, both because the core outshines them
//     by orders of magnitude: light entering the globe from the room, and rays
//     that leave the medium without scattering (well under 1% of entry rays at
//     this density) which would otherwise carry on and see the room behind it.
//
//     Only delta chains arrive here (camera rays, and the globe's reflection in
//     the mirror or its image through the glass). Anything that scattered
//     diffusely already had the globe estimated by globeLight, so letting it in
//     would double count -- the same rule the two emitters follow.
#define GLOBE_BOUNCE 8

vec3 globeWalk(vec3 p, vec3 rd, float time) {
    vec3 core = coreRadiance(time);
    vec3 L = vec3(0.0);
    vec3 thr = vec3(1.0);
    bool straight = true;      // still on the unscattered entry segment
    for (int i = 0; i < GLOBE_BOUNCE; i++) {
        float tOut = sphereFar(p, rd, GLOBE_C, GLOBE_R);
        float tCore = sphereNear(p, rd, GLOBE_C, CORE_R);
        float s = -log(max(1.0 - rnd(), 1e-6)) / SIGMA_T;
        if (tCore > 1e-5 && tCore < min(s, tOut)) {
            // reached the filament: opaque, so the walk ends here either way.
            // Its emission counts only on the entry segment, for the same
            // reason as above -- afterwards NEE is already carrying it.
            if (straight) L += thr * core;
            return L;
        }
        if (s >= tOut) break;      // escaped without scattering (rare, this dense)
        p += rd * s;

        // NEE to the filament. Nothing else can occlude inside the globe and the
        // medium is homogeneous, so the transmittance along the connection is
        // just exp(-sigma_t * distance) -- no ratio tracking needed.
        vec3 oc = GLOBE_C - p;
        float d2 = dot(oc, oc);
        float cr2 = CORE_R * CORE_R;
        if (d2 > cr2) {
            float cosMax = sqrt(max(1.0 - cr2 / d2, 0.0));
            vec3 l = sampleCone(oc * inversesqrt(d2), cosMax, rnd2());
            float b = -dot(oc, l);
            float disc = b * b - (d2 - cr2);
            if (disc > 0.0) {
                float tc = -b - sqrt(disc);
                if (tc > 1e-5)
                    L += thr * core * (TAU * (1.0 - cosMax))   // 1 / pdf
                         * (0.25 / PI)                          // isotropic phase
                         * exp(-SIGMA_T * tc);                  // transmittance
            }
        }
        rd = uniformSphereDir(rnd2());
        straight = false;
        if (i >= 2) {                          // russian roulette
            if (rnd() > 0.7) break;
            thr /= 0.7;
        }
    }
    return L;
}

vec3 tracePath(vec2 fragCoord, vec3 res, vec2 ang, float time, int sIdx) {
    vec2 jit = rnd2() - 0.5;
    vec3 ro, rd;
    camRay(fragCoord + jit, res, ang, ro, rd);

    vec3 L = vec3(0.0);        // still on the delta chain: what the eye sees
    vec3 Lind = vec3(0.0);     // gathered after a diffuse bounce: clamped below
    vec3 thr = vec3(1.0);
    bool inGlass    = false;    // inside the absorbing dielectric
    bool prevDelta  = true;     // the event before this one was a delta one
    bool afterDiff  = false;    // a non-delta bounce has happened
    bool firstIface = true;     // stratify the first Fresnel branch
    float pathLen = 0.0;        // unfolded path length, for the filter footprint
    float pixAng = pixelAngle(res);

    for (int b = 0; b < MAX_BOUNCE; b++) {
        Hit h = sceneHit(ro, rd, FAR);
        if (h.mat == MAT_NONE) break;             // ray left the room: no sky
        if (inGlass) thr *= exp(-GLASS_ABS * h.t);
        vec3 p = ro + rd * h.t;
        pathLen += h.t;

        if (h.mat == MAT_LIGHT || h.mat == MAT_GLOBE) {
            // Emission counts only when the previous event was DELTA. Every
            // non-delta vertex already estimated both emitters with a shadow
            // ray, so counting a hit there too would double it. But a shadow ray
            // travels in a straight line and cannot bend through the glass -
            // those paths are missing from NEE entirely, and they are exactly
            // the caustic. Hence the flag rather than a blanket rule.
            if (prevDelta) {
                // the globe is not a surface but a volume: walk it
                vec3 Le = h.mat == MAT_GLOBE ? globeWalk(p, rd, time) : LIGHT_E;
                if (afterDiff) Lind += thr * Le;
                else           L    += thr * Le;
            }
            break;                                // both emitters end the path
        }

        if (h.mat == MAT_GLASS) {
            vec3 nF = h.n;                        // normal, incident side
            float eta = IOR_GLASS;                // n_t / n_i
            if (inGlass) { nF = -h.n; eta = 1.0 / IOR_GLASS; }
            float cosI = clamp(dot(-rd, nF), 0.0, 1.0);
            float Fr = frDielectric(cosI, eta);
            // Stratified across the frame's samples at the FIRST interface: the
            // reflect-or-refract coin flip is the loudest term on a glass ball,
            // and taking one of each per pixel silences most of it. Deeper
            // interfaces are random again - stratifying those would correlate
            // branches that have already diverged.
            float u = rnd();
            if (firstIface) u = (float(sIdx) + u) / float(SPP);
            firstIface = false;
            if (u < Fr) {                         // Fr is exactly 1.0 on TIR
                rd = reflect(rd, nF);
                ro = p + nF * 2e-4;
            } else {
                rd = normalize(refract(rd, nF, 1.0 / eta));
                ro = p - nF * 2e-4;
                inGlass = !inGlass;
            }
            prevDelta = true;
            continue;
        }

        if (h.mat == MAT_METAL) {
            // smooth conductor: one delta reflection, Schlick-tinted
            float cosI = clamp(dot(-rd, h.n), 0.0, 1.0);
            thr *= METAL_F0 + (1.0 - METAL_F0) * pow(1.0 - cosI, 5.0);
            rd = reflect(rd, h.n);
            ro = p + h.n * 2e-4;
            prevDelta = true;
            continue;
        }

        // --- matte / glazed surface -----------------------------------------
        // Footprint of this pixel on the surface (metres): grows with path
        // length and with grazing angle, which is what the checker filter needs.
        float fw = pathLen * pixAng / max(abs(dot(h.n, rd)), 0.45);
        vec3 alb; float rough;
        materialAt(h.mat, p, fw, alb, rough);
        vec3 v = -rd;

        vec3 direct = directLight(p, h.n, v, alb, rough)
                    + globeLight(p, h.n, v, alb, rough, time);
        if (afterDiff) Lind += thr * direct;
        else           L    += thr * direct;

        // next direction: glazed specular lobe or diffuse, chosen with a
        // probability that tracks the Fresnel weight so the estimator is not
        // fighting its own choice
        float Fv = 0.04 + 0.96 * pow(1.0 - clamp(dot(h.n, v), 0.0, 1.0), 5.0);
        float pSpec = clamp(Fv * 1.25, 0.05, 0.90);
        if (rnd() < pSpec) {
            vec3 hv = sampleGGX(h.n, v, rough, rnd2());
            vec3 l = reflect(rd, hv);
            if (dot(l, h.n) <= 0.0) break;
            float VoH = clamp(dot(v, hv), 0.0, 1.0);
            float F = 0.04 + 0.96 * pow(1.0 - VoH, 5.0);
            float a2 = rough * rough; a2 *= a2;
            float NoL = clamp(dot(h.n, l), 1e-4, 1.0);
            // VNDF sampling weight with separable Smith: F * G1(L)
            float G1L = 2.0 * NoL / (NoL + sqrt(a2 + (1.0 - a2) * NoL * NoL));
            thr *= F * G1L / pSpec;
            rd = l;
        } else {
            thr *= alb * (1.0 - Fv) / (1.0 - pSpec);
            rd = cosineDir(h.n, rnd2());
        }
        ro = p + h.n * 2e-4;
        prevDelta = false;
        afterDiff = true;

        // russian roulette. A closed room has no escape hatch, so without this
        // every path would run the full bounce budget.
        if (b >= 2) {
            float q = clamp(max(thr.x, max(thr.y, thr.z)), 0.05, 0.95);
            if (rnd() > q) break;
            thr /= q;
        }
    }

    // Guard NaN / inf and clip fireflies. Demodulation happens once per pixel in
    // mainImage, not here: it has to use the albedo of the PSR surface the
    // G-buffer reports, which is the one the Image pass will multiply back.
    if (!(L.x >= 0.0) || !(L.y >= 0.0) || !(L.z >= 0.0)) L = vec3(0.0);
    if (!(Lind.x >= 0.0) || !(Lind.y >= 0.0) || !(Lind.z >= 0.0)) Lind = vec3(0.0);
    return min(L, vec3(FIREFLY_MAX)) + min(Lind, vec3(CLAMP_INDIRECT));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    seedRNG(fragCoord, iFrame);

    // Camera state pixel. The orbit angles are integrated here and published to
    // every other pass, together with the previous frame's angles - Buffer B
    // needs those to reproject, and a mouse-driven camera cannot be recovered
    // from iTime the way a scripted one could.
    vec4 cam = stepCam(camState(iChannel0), iMouse, iResolution, iTimeDelta, iFrame);
    if (ivec2(fragCoord) == ivec2(0)) {           // state pixel, no tracing
        fragColor = cam;
        return;
    }

    // One deterministic centre-ray walk gives the G-buffer for all four
    // denoiser passes (see primaryHitFull: it follows the specular chain), and
    // with it the albedo to divide out.
    float tP; vec3 nP; int matP; vec3 posP, dirP;
    primaryHitFull(fragCoord, iResolution, cam.xy, tP, nP, matP, posP, dirP);
    float fwP = tP * pixelAngle(iResolution) / max(abs(dot(nP, dirP)), 0.45);
    vec3 albP = psrAlbedo(matP, posP, fwP);

    vec3 L = vec3(0.0);
    for (int s = 0; s < SPP; s++)
        L += tracePath(fragCoord, iResolution, cam.xy, iTime, s);
    L /= float(SPP);

    // Demodulate: the denoiser filters irradiance, and the Image pass puts the
    // albedo back with a supersampled lookup, which is what keeps material
    // edges crisp through a blur this wide.
    fragColor = vec4(L / max(albP, vec3(0.02)), packGbuf(tP, nP, matP));
}
