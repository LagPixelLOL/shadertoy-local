// ============================================================================
//  Image -- put the sky behind the cloud, add the shafts, expose and tonemap.
//
//  iChannel0: Buffer B (rgb = scattered light, a = transmittance), linear
//
//  Buffers mean this shader has to be run forward from frame 0, which the
//  renderer does by default (--precharge all):
//      shadertoy render -C examples/08-cumulonimbus --frame 120
//  The temporal filter has converged after about 30 frames, so a cheaper render
//  of the same picture is:
//      shadertoy render -C examples/08-cumulonimbus --frame 120 --precharge 32
// ============================================================================

// Crepuscular rays. A radial blur of the cloud's transmittance toward the sun
// is the cheapest thing that gets this right, and it is not a cheat: the value
// being smeared really is the fraction of the beam that survives the cloud
// along that line, which is exactly what the air between the storm and the
// camera is being lit by. Sixteen taps is enough because the thing being
// sampled has already been through a 16-frame temporal filter and is smooth.
#define SHAFT_STEPS 16
#define SHAFT_DECAY 0.94
#define SHAFT_STRENGTH 0.055

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec3 sd = sunDirection(iTime);
    vec2 ang = cameraAngles(iTime);
    vec3 ro, rd;
    cameraRay(fragCoord, iResolution, ang, ro, rd);

    vec4 cloud = texture(iChannel0, fragCoord / iResolution.xy);

    // Everything behind the storm: sky, and the sun if it is in this pixel. The
    // cloud's own transmittance is what hides it, which is why the disc dims
    // and reddens as an anvil edge drifts across it without any special case.
    vec3 background = skyRadiance(rd, sd) + sunDisc(rd, sd);
    vec3 col = background * cloud.a + cloud.rgb;

    vec2 sunCoord = cameraProject(sd, iResolution, ang);
    if (sunCoord.x > -1e8 && sd.y > 0.0) {
        vec2 uv = fragCoord / iResolution.xy;
        vec2 delta = (sunCoord / iResolution.xy - uv) / float(SHAFT_STEPS);
        float shaft = 0.0, weight = 1.0, total = 0.0;
        for (int i = 0; i < SHAFT_STEPS; i++) {
            uv += delta;
            // Taps that leave the frame are dropped, not clamped. Clamping
            // replicates the edge texel along the whole remaining tail, which
            // draws a hard horizontal streak out of the side of the frame --
            // the one artefact a radial blur is guaranteed to produce if you
            // let it.
            float inside = float(all(greaterThanEqual(uv, vec2(0.0))) &&
                                 all(lessThanEqual(uv, vec2(1.0))));
            shaft += mix(1.0, texture(iChannel0, uv).a, inside) * weight;
            total += weight;
            weight *= SHAFT_DECAY;
        }
        // Restricted to the forward lobe. Shafts are forward-scattered light,
        // so they belong near the sun and nowhere else; without this the blur
        // lays a grey veil across the whole frame.
        float lobe = pow(max(dot(rd, sd), 0.0), 14.0);
        col += SUN_RADIANCE * sunTransmittance(sd) *
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
    float night = nightness(sd, rdC);
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
