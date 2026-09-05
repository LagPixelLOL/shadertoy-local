// ============================================================================
//  Image -- put the sky behind the cloud, add the shafts, expose and tonemap.
//
//  iChannel0: Buffer B (rgb = scattered light, a = transmittance), linear
//  iChannel1: keyboard -- S toggles sun/moon; iMouse steers the sun
//
//  Buffers mean this shader has to be run forward from frame 0, which the
//  renderer does by default (--precharge all):
//      shadertoy render -C examples/08-cumulus --frame 120
//  For a quicker preview with partial temporal warm-up (not identical history):
//      shadertoy render -C examples/08-cumulus --frame 120 --precharge 32
// ============================================================================

// Screen-space approximation to crepuscular rays. It uses view transmittance,
// not a 3D shadow volume, so off-screen occluders cannot contribute.
#define SHAFT_STEPS 24
#define SHAFT_DECAY 0.96
#define SHAFT_STRENGTH 0.055

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Same lighting state as Buffer A: see setupLighting.
    setupLighting(iMouse, iResolution, texelFetch(iChannel1, ivec2(83, 2), 0).x);
    vec3 sd = sunDirection();
    vec2 ang = cameraAngles(iTime);
    vec3 ro, rd;
    cameraRay(fragCoord, iResolution, ang, ro, rd);

    vec4 cloud = texture(iChannel0, fragCoord / iResolution.xy);
    // Buffer B reserves this sky-border texel for temporal lighting state.
    if (all(lessThan(fragCoord, vec2(1.0)))) cloud = vec4(0.0, 0.0, 0.0, 1.0);

    // Cloud transmittance occludes the sky and sun. SKY_GAIN calibrates the
    // approximate atmosphere against the cloud's multiple-scattering model.
    vec3 background = (skyRadiance(rd, sd) + sunDisc(rd, sd)) * SKY_GAIN;
    vec3 col = background * cloud.a + cloud.rgb;

    vec2 sunCoord = cameraProject(sd, iResolution, ang);
    if (sunCoord.x > -1e8 && sd.y > 0.0) {
        vec2 uv = fragCoord / iResolution.xy;
        vec2 delta = (sunCoord / iResolution.xy - uv) / float(SHAFT_STEPS);

        // The sampling phase is jittered per pixel, and the jitter is
        // *static* -- no frame term. For a pixel half a frame from the sun
        // each tap lands fifty pixels from the last, and on a regular
        // progression every silhouette edge is echoed at each of those
        // offsets: the rays come out stepped, a fan of ghost edges instead
        // of a smooth beam. Sliding the whole progression by a per-pixel
        // fraction of one step decorrelates neighbouring pixels' echoes
        // into noise the eye reads as a smooth gradient. A frame-varying
        // jitter would average even smoother, but this pass runs after the
        // temporal filter, so anything time-dependent here shimmers.
        uv += delta * ign(fragCoord);

        float shaft = 0.0, weight = 1.0, total = 0.0;
        for (int i = 0; i < SHAFT_STEPS; i++) {
            uv += delta;
            // Off-screen taps assume clear sky, rather than extending the edge
            // texel into a false streak. Keep them in the normalization weight.
            float inside = float(all(greaterThanEqual(uv, vec2(0.0))) &&
                                 all(lessThanEqual(uv, vec2(1.0))));
            vec2 sampleUV = max(uv, 1.5 / iResolution.xy);
            shaft += mix(1.0, texture(iChannel0, sampleUV).a, inside) * weight;
            total += weight;
            weight *= SHAFT_DECAY;
        }
        // Restricted to the forward lobe. Shafts are forward-scattered light,
        // so they belong near the sun and nowhere else; without this the blur
        // lays a grey veil across the whole frame.
        float lobe = pow(max(dot(rd, sd), 0.0), 14.0);
        col += g_sunRadiance * sunTransmittance(sd) *
               (shaft / total) * lobe * SHAFT_STRENGTH;
    }

    // Exposure is a camera property, so it is computed once for the frame
    // centre, not per pixel; see sceneExposure. The Purkinje shift is what
    // keeps a moonlit version of this scene from being day with the lights
    // dimmed: rod vision holds no colour, so as the key light falls stops
    // below the calibration sun the frame slides toward dim blue-grey
    // monochrome before the tone curve ever sees it.
    vec3 roC, rdC;
    cameraRay(iResolution.xy * 0.5, iResolution, ang, roC, rdC);
    float night = nightness();
    // Fixed daylight camera balance removes the high-sun yellow cast without
    // neutralizing the warmer spectrum at sunset or adding blue to moonlight.
    col *= mix(vec3(0.88, 1.0, 1.12), vec3(1.0), night);
    if (night > 0.0) {
        float l = dot(col, vec3(0.2126, 0.7152, 0.0722));
        col = mix(col, l * vec3(0.62, 0.90, 1.45), 0.85 * night);
    }

    col = tonemap(col * sceneExposure(sd, rdC), cloud.a);
    col = pow(col, vec3(0.4545));   // linear -> sRGB

    // The sky is a smooth gradient over hundreds of pixels, which is the exact
    // case 8-bit quantisation turns into visible bands.
    col += (hash12(fragCoord + fract(iTime) * 113.1) - 0.5) * (1.4 / 255.0);

    fragColor = vec4(col, 1.0);
}
