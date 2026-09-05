// Full-resolution cumulus raymarch.
// iChannel0: RGBA Noise Medium, linear/repeat; iChannel1: Keyboard.
// Output: scattered radiance (rgb), view transmittance (a).
#define MAX_STEPS 512
#define MIN_TRANSMITTANCE 0.005
const vec3 BETA_AERIAL = BETA_R * 0.50 + vec3(1.2e-3);

// Resolve nearby shadow boundaries before increasing the step. Capping distant
// intervals avoids bright leaks from point-sampling an entire kilometre of cloud.
float lightMarch(sampler2D tex, vec3 p, vec3 sd, float time, float jitter) {
    float od = 0.0, dist = 0.0, ds = 0.015;
    float lightEntry, lightExit;
    if (!boxRange(p, sd, lightEntry, lightExit)) return 0.0;
    for (int i = 0; i < 28; i++) {
        if (dist >= lightExit) break;
        ds = min(ds, lightExit - dist);
        vec3 q = p + sd * (dist + ds * jitter);
        float sdf = cloudEnvelope(q, time);
        od += cloudDensity(tex, q, time, sdf) * ds;
        dist += ds;
        ds = min(ds * 1.4, 0.42);
    }
    return od * EXTINCTION;
}

float ambientMarch(sampler2D tex, vec3 p, float time, float jitter) {
    float od = 0.0, ds = 0.06, dist = 0.0;
    for (int i = 0; i < 6; i++) {
        vec3 q = p + vec3(0.0, dist + jitter * ds, 0.0);
        float sdf = cloudEnvelope(q, time);
        od += cloudDensity(tex, q, time, sdf) * ds;
        dist += ds;
        ds *= 2.0;
    }
    return od * EXTINCTION;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    setupLighting(iMouse, iResolution, texelFetch(iChannel1, ivec2(83, 2), 0).x);
    vec3 sd = sunDirection();
    vec2 ang = cameraAngles(iTime);
    vec3 ro, rd;
    cameraRay(fragCoord + haltonJitter(iFrame), iResolution, ang, ro, rd);
    vec3 scattered = vec3(0.0);
    float transmittance = 1.0;
    float t0, t1;
    if (boxRange(ro, rd, t0, t1)) {
        vec3 sunCol = g_sunRadiance * sunTransmittance(sd);
        const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);
        vec3 keyCol = KEY_RADIANCE * sunTransmittance(keySunDirection());
        vec3 nowCol = KEY_RADIANCE * sunTransmittance(sd);
        float refill = clamp(dot(nowCol, LUMA) / dot(keyCol, LUMA), 0.0, 1.0);
        vec3 roC, rdC;
        cameraRay(iResolution.xy * 0.5, iResolution, ang, roC, rdC);
        float gNow = 0.5 + 0.5 * dot(sd, -rdC);
        float gKey = 0.5 + 0.5 * dot(keySunDirection(), -rdC);
        refill *= clamp(pow(max(gNow, 1e-3) / gKey, 0.45), 0.30, 1.2);
        vec3 skyLight = mix(skyRadiance(vec3(0.0, 1.0, 0.0), sd),
                            skyRadiance(normalize(vec3(sd.x, 0.22, sd.z)), sd), 0.15) * 2.5;
        vec3 haze = skyRadiance(rd, sd);
        vec3 backFill = skyRadiance(normalize(vec3(0.0, 0.40, -0.92)), sd) * 0.5;
        backFill = mix(vec3(dot(backFill, LUMA)), backFill, 0.15);
        float mu = dot(rd, sd);
        float offset = fract(ign(fragCoord) + float(iFrame) * 0.6180339887);
        // Independent light/view phases avoid a structured bias in thin fringes.
        float lightJitter = fract(hash12(fragCoord + 19.19) + float(iFrame) * 0.754877666);
        float t = max(t0, 0.0);
        float pixelAngle = 2.0 / (iResolution.y * FOCAL);
        bool entered = false;
        for (int i = 0; i < MAX_STEPS; i++) {
            if (t > t1 || transmittance < MIN_TRANSMITTANCE) break;
            vec3 p = ro + rd * t;
            float sdf = cloudEnvelope(p, iTime);
            float minStep = mix(0.010, 0.006, crownCondensation(p.y));
            float ds = clamp(t * pixelAngle, minStep, 0.04);
            if (sdf > INFLATE) {
                t += max((sdf - INFLATE) * 0.9, ds);
                entered = false;
                continue;
            }
            if (!entered) {
                entered = true;
                t += offset * ds;
                continue;
            }
            float density = cloudDensity(iChannel0, p, iTime, sdf);
            if (density > 0.001) {
                float opticalDepth = lightMarch(iChannel0, p, sd, iTime, lightJitter);
                float skyDepth = ambientMarch(iChannel0, p, iTime, lightJitter);
                vec3 source = sunScatter(opticalDepth, mu, sunCol) +
                              ambientScatter(skyLight, skyDepth, opticalDepth, refill) +
                              backFill * transmittance;
                source *= ALBEDO;
                // Beer-Lambert integration is independent of the segment length.
                float stepT = exp(-density * EXTINCTION * ds);
                vec3 clear = exp(-BETA_AERIAL * t);
                vec3 segment = (source * clear + haze * (1.0 - clear)) * (1.0 - stepT);
                scattered += transmittance * segment;
                transmittance *= stepT;
            }
            t += ds;
        }
    }
    // An opaque early-out must not leak a bright sun disc through the core.
    float alpha = max(0.0, (transmittance - MIN_TRANSMITTANCE) / (1.0 - MIN_TRANSMITTANCE));
    fragColor = vec4(scattered, alpha);
}
