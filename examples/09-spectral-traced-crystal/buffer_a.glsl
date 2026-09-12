// Continuous-wavelength path tracing, explicit beam connections, and single-scatter
// haze. Pixels (0,0) and (1,0) store crystal/input state; other alphas pack the G-buffer.
// Each path transports scalar radiance at one wavelength; output RGB slots hold XYZ.
#define SPP 1
#define SPECTRAL_PACKET 1
#define MAX_BOUNCE 64

// Clip a camera segment into the bar's projected aperture. All mappings below
// are affine, so integrating beam visibility needs no noisy volume ray march.
bool clipAperture(vec3 o, vec3 d, inout float lo, inout float hi) {
    if (!clipPlane(o, d, vec4(0, 1, 0, ROOM.y), lo, hi)) return false;
    if (!clipPlane(o, d, vec4(0, -1, 0, 0), lo, hi)) return false;
    if (!clipPlane(o, d, vec4(0, 0, 1, LIGHT_HALF_W), lo, hi)) return false;
    return clipPlane(o, d, vec4(0, 0, -1, LIGHT_HALF_W), lo, hi);
}

float beamWeight(float lo, float hi, bool volume, vec3 n, vec3 v, vec3 l,
                 float alb, float rough, float opticalBase,
                 float opticalSlope) {
    if (!volume)
        return evalBRDF(n, v, -l, alb, rough) * max(dot(n, -l), 0.0)
               * exp(-opticalBase);
    // Both source and camera optical distances vary along the segment.
    float k = HAZE + opticalSlope;
    if (abs(k) < 1e-5)
        return HAZE * (hi - lo) * exp(-opticalBase - k * 0.5 * (lo + hi)) / (4.0 * PI);
    return HAZE * (exp(-opticalBase - k * lo) - exp(-opticalBase - k * hi))
           / (4.0 * PI * k);
}

float beamLight(vec3 ro, vec3 rd, float tmax, bool volume, vec3 n, vec3 v,
                float alb, float rough, float ior, float absorption, vec3 incident) {
    float total = 0.0;
    float lo = 0.0, hi = tmax;
    vec3 src = ro - incident * ((ro.x + ROOM.x) / incident.x);
    vec3 srcD = rd - incident * (rd.x / incident.x);
    if (clipAperture(src, srcD, lo, hi)) {
        float opticalBase = HAZE * (ro.x + ROOM.x) / incident.x;
        float opticalSlope = HAZE * rd.x / incident.x;
        // Remove the solid's shadow from the otherwise continuous light sheet.
        float shadowLo = lo, shadowHi = hi;
        bool shadow = true;
        // Projection onto each illuminated face, clipped against the whole solid.
        float blocked = 0.0;
        for (int i = 0; i < 15; i++) {
            vec4 pl = crystalPlane(i);
            float ci = dot(incident, pl.xyz);
            if (ci >= -1e-5) continue;
            float s = (dot(ro - CRYSTAL_C, pl.xyz) - pl.w) / ci;
            float sd = dot(rd, pl.xyz) / ci;
            vec3 q = ro - incident * s, qd = rd - incident * sd;
            shadowLo = lo; shadowHi = hi;
            shadow = clipPlane(vec3(-s, 0, 0), vec3(-sd, 0, 0),
                               vec4(1, 0, 0, 0), shadowLo, shadowHi);
            if (shadow && clipCrystal(q, qd, i, shadowLo, shadowHi))
                blocked += beamWeight(shadowLo, shadowHi, volume, n, v,
                                      incident, alb, rough, opticalBase, opticalSlope);
        }
        total += max(0.0, beamWeight(lo, hi, volume, n, v, incident,
                                    alb, rough, opticalBase, opticalSlope) - blocked);
    }

    // Enumerate planar entry/exit pairs. Reverse-projecting through both planes
    // finds the exact footprint of each refracted beam, including pyramid tips.
    // This explicitly samples paths a straight NEE shadow ray cannot connect.
    for (int entry = 0; entry < 15; entry++) {
        vec4 p1 = crystalPlane(entry);
        float ci = -dot(incident, p1.xyz);
        if (ci <= 1e-5) continue;
        vec3 inside = refract(incident, p1.xyz, 1.0 / ior);
        float ct = -dot(inside, p1.xyz);
        float f1 = 1.0 - frDielectric(ci, ior);
        for (int exitFace = 0; exitFace < 15; exitFace++) {
            vec4 p2 = crystalPlane(exitFace);
            float c2 = dot(inside, p2.xyz);
            if (c2 <= 1e-5) continue;
            vec3 outgoing = refract(inside, -p2.xyz, ior);
            if (dot(outgoing, outgoing) < 0.5) continue; // total internal reflection
            float co = dot(outgoing, p2.xyz);
            if (!volume && dot(n, -outgoing) <= 0.0) continue;

            float s2 = (dot(ro - CRYSTAL_C, p2.xyz) - p2.w) / co;
            float d2 = dot(rd, p2.xyz) / co;
            vec3 q2 = ro - outgoing * s2, q2d = rd - outgoing * d2;
            lo = 0.0; hi = tmax;
            if (!clipPlane(vec3(-s2, 0, 0), vec3(-d2, 0, 0),
                           vec4(1, 0, 0, 0), lo, hi)) continue;
            float s1 = (dot(q2 - CRYSTAL_C, p1.xyz) - p1.w) / (-ct);
            float d1 = dot(q2d, p1.xyz) / (-ct);
            vec3 q1 = q2 - inside * s1, q1d = q2d - inside * d1;
            if (!clipPlane(vec3(-s1, 0, 0), vec3(-d1, 0, 0),
                           vec4(1, 0, 0, 0), lo, hi)) continue;
            src = q1 - incident * ((q1.x + ROOM.x) / incident.x);
            srcD = q1d - incident * (q1d.x / incident.x);
            // The narrow slit rejects most candidates. Intersect it before the
            // two fifteen-plane face footprints; the final interval is unchanged.
            if (!clipAperture(src, srcD, lo, hi)) continue;
            if (!clipCrystal(q2, q2d, exitFace, lo, hi)) continue;
            if (!clipCrystal(q1, q1d, entry, lo, hi)) continue;

            float opticalBase = absorption * s1
                              + HAZE * (s2 + (q1.x + ROOM.x) / incident.x);
            float opticalSlope = absorption * d1
                               + HAZE * (d2 + q1d.x / incident.x);
            float f2 = 1.0 - frDielectric(c2, 1.0 / ior);
            float jacobian = (ci * c2) / max(ct * co, 1e-5);
            total += beamWeight(lo, hi, volume, n, v, outgoing, alb, rough,
                                opticalBase, opticalSlope) * f1 * f2 * jacobian;
        }
    }
    return total * BEAM_E;
}

float diffuseBar(vec3 p, vec3 n, vec3 v, float alb, float rough) {
    vec3 d = lightPoint(rnd2()) - p;
    float r2 = dot(d, d), r = sqrt(r2);
    d /= max(r, 1e-5);
    float nl = dot(n, d), cl = -d.x;
    if (nl <= 0.0 || cl <= 0.0) return 0.0;
    if (sceneOccluded(p + n * 2e-4, d, r - 2e-3)) return 0.0;
    return evalBRDF(n, v, d, alb, rough) * nl * LIGHT_E
           * (cl * LIGHT_A / max(r2, 1e-6)) * exp(-HAZE * r);
}

float tracePath(vec2 fc, vec3 res, float wavelength) {
    float ior = glassIOR(wavelength);
    float absorption = glassAbsorption(wavelength);
    vec3 ro, rd; camRay(fc + rnd2() - 0.5, res, ro, rd);
    vec3 incident = sampleCone(vec3(1, 0, 0), cos(BEAM_ANGLE), rnd2());
    float L = 0.0, thr = 1.0;
    bool inGlass = false, prevDelta = true, afterDiff = false;
    float pathLen = 0.0;

    for (int b = 0; b < MAX_BOUNCE; b++) {
        Hit h = sceneHit(ro, rd, FAR);
        if (h.mat == MAT_NONE) break;
        if (inGlass) thr *= exp(-absorption * h.t);
        else {
            if (!afterDiff)
                L += thr * beamLight(ro, rd, h.t, true, vec3(0), -rd,
                                     1.0, 0.5, ior, absorption, incident);
            thr *= exp(-HAZE * h.t);
        }
        vec3 p = ro + rd * h.t;
        pathLen += h.t;
        if (h.mat == MAT_LIGHT) {
            // The broad, weak diffuser is estimated separately from the narrow
            // beam; direct/specular views still see the actual bright aperture.
            if (prevDelta) L += thr * LIGHT_E;
            if (!afterDiff && -rd.x > cos(BEAM_ANGLE))
                L += thr * BEAM_E / (TAU * (1.0 - cos(BEAM_ANGLE)));
            break;
        }
        if (h.mat == MAT_GLASS) {
            vec3 nf = inGlass ? -h.n : h.n;
            float eta = inGlass ? 1.0 / ior : ior;
            float fr = frDielectric(dot(-rd, nf), eta);
            if (fr >= 1.0 || rnd() < fr) { rd = reflect(rd, nf); ro = p + nf * 2e-4; }
            else {
                rd = normalize(refract(rd, nf, 1.0 / eta));
                ro = p - nf * 2e-4; inGlass = !inGlass;
            }
            prevDelta = true;
            continue;
        }
        vec2 fw = vec2(pathLen * pixelAngle(res) / max(abs(dot(h.n, rd)), 0.45));
        float alb, rough; materialAt(h.mat, p, fw, alb, rough);
        vec3 v = -rd;
        float direct = diffuseBar(p, h.n, v, alb, rough)
                     + beamLight(p + h.n * 2e-4, vec3(0), 0.0, false,
                                 h.n, v, alb, rough, ior, absorption, incident);
        L += thr * direct;
        float fv = 0.04 + 0.96 * pow(1.0 - clamp(dot(h.n, v), 0.0, 1.0), 5.0);
        float ps = clamp(fv * 1.25, 0.05, 0.9);
        if (rnd() < ps) {
            vec3 hv = sampleGGX(h.n, v, rough, rnd2());
            vec3 l = reflect(rd, hv);
            if (dot(l, h.n) <= 0.0) break;
            float f = 0.04 + 0.96 * pow(1.0 - clamp(dot(v, hv), 0.0, 1.0), 5.0);
            float a2 = pow(rough, 4.0), nl = max(dot(h.n, l), 1e-4);
            float nv = max(dot(h.n, v), 1e-4);
            float g1v = 2.0 * nv / (nv + sqrt(a2 + (1.0 - a2) * nv * nv));
            thr *= f * 4.0 * nv * nl * G_smith(nv, nl, a2) / (g1v * ps);
            rd = l;
        } else {
            rd = cosineDir(h.n, rnd2());
            float vh = clamp(dot(v, normalize(v + rd)), 0.0, 1.0);
            float f = 0.04 + 0.96 * pow(1.0 - vh, 5.0);
            thr *= alb * (1.0 - f) / (1.0 - ps);
        }
        ro = p + h.n * 2e-4;
        prevDelta = false; afterDiff = true;
        if (b >= 3) {
            float q = clamp(thr, 0.2, 0.9);
            if (rnd() > q) break;
            thr /= q;
        }
    }
    // Every contribution comes from the same D65 bar, so its spectral power
    // factors out. Do not clamp bright wavelengths before spectral integration.
    return L >= 0.0 && !isinf(L) ? L * illuminantD65(wavelength) : 0.0;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    seedRNG(fragCoord, iFrame);
    bool roll = texelFetch(iChannel1, ivec2(16, 0), 0).r > 0.5; // Shift
    CrystalState pose = stepCrystal(crystalState(iChannel0), iMouse, iResolution, roll, iFrame);
    if (ivec2(fragCoord) == ivec2(0, 0)) { fragColor = pose.rotation; return; }
    if (ivec2(fragCoord) == ivec2(1, 0)) { fragColor = pose.controls; return; }
    setCrystalRotation(pose.rotation);
    float t; vec3 n; int mat; vec3 p, d;
    primaryHitFull(fragCoord, iResolution, t, n, mat, p, d);
    vec2 fw = psrFootprint(fragCoord, iResolution, t, n, mat, p, d);
    float alb = psrAlbedo(mat, p, fw);
    vec3 xyz = vec3(0);
    for (int group = 0; group < SPP / SPECTRAL_PACKET; group++) {
        // Stratify the full wavelength interval, not three RGB bands. Shared
        // path randomness within each packet reduces chromatic noise, but each
        // wavelength follows its own refracted path and Fresnel decisions.
        float u = (float(group) + rnd()) / float(SPP);
        uvec4 state = _rs;
        for (int k = 0; k < SPECTRAL_PACKET; k++) {
            _rs = state;
            float wavelength = mix(LAMBDA_MIN, LAMBDA_MAX, u + float(k) / float(SPECTRAL_PACKET));
            float radiance = tracePath(fragCoord, iResolution, wavelength);
            xyz += radiance * cieXYZ(wavelength) / (WAVELENGTH_PDF * D65_Y_INTEGRAL);
        }
    }
    fragColor = vec4(xyz / (float(SPP) * max(alb, 0.02)), packGbuf(t, n, mat));
}
