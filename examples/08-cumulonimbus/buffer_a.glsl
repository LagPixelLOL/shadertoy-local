// ============================================================================
//  Buffer A -- the volumetric march. One jittered sample per pixel per frame;
//  Buffer B averages sixteen of them.
//
//  iChannel0: RGBA Noise Medium (256), linear/repeat -- the value-noise tile
//  output:    rgb = light scattered toward the camera (aerial perspective
//             already applied), a = transmittance of the whole ray
//
//  Compositing is left to the Image pass, which multiplies the sky by that
//  alpha and adds the rgb. Keeping the sky out of here means it stays pixel
//  sharp: only the cloud, which is the noisy part, goes through the temporal
//  filter.
//
//  Cost, measured on an RTX PRO 6000 at frame 300, whole pipeline per frame:
//
//      640x360     1.4 ms
//      1280x720    4.5 ms
//      1920x1080   8.9 ms
//
//  Most of that is the erosion fBm in cloudDensity -- three fractals of two
//  texture fetches per octave, evaluated again for every light-march and
//  ambient-march sample -- and it is what the detail is made of, so it is
//  not recoverable by tuning.
//
//  The one optimisation that carries this is the transmittance cutoff: a ray
//  that has hit the opaque interior stops rather than marching out the far
//  side of a cloud nothing can be seen through. Everything else is load
//  balancing: the bounding slab makes the empty sky corners cost one test, the
//  sphere-trace jump crosses the gap between the camera and the crown without
//  sampling noise (worth little in this tight framing -- measured within noise
//  of zero -- but it is also free), and the step length tracks the pixel
//  footprint so the render resolution sets the sample density.
// ============================================================================

// Primary march. The cap is a safety net, not the working step count: with the
// cutoff below, a ray into the middle of the crown terminates within a few
// dozen steps of entering cloud. What the cap actually bounds is the worst
// case, a silhouette-grazing ray that spends its whole length in low-density
// wisp.
#define MAX_STEPS 256

// Stop when the cloud in front has absorbed 99% of what is behind it. The
// remaining 1% is below the dither floor of an 8-bit image.
#define MIN_TRANSMITTANCE 0.01

// Light march toward the sun: five samples on a geometric progression, so the
// near field where the gradient is steep gets resolution and the far field,
// which only contributes bulk shadowing, gets reach. Total 4.96 km, which is
// wider than a turret.
#define LIGHT_STEPS 6
#define LIGHT_STEP0 0.05
#define LIGHT_GROWTH 2.0

// Below this the sample contributes less than the 8-bit floor, so both the
// light march and the scattering evaluation are skipped and the previous
// optical depth is carried forward. Density is continuous along a ray, so the
// carried value is a good estimate of the one not computed -- and it is only
// ever used where the weight is negligible anyway.
#define LIGHT_MARCH_CUTOFF 0.015

// How much of the fine erosion the light march sees. Not zero, which is what it
// was: with the shadowing computed from the coarse field only, every fine bump
// was lit as if it were flat, and the erosion rendered as bright pillows
// pasted onto the shading. A bump has to shade its own far side or it is not
// a bump. Most rather than all, because the six samples cannot resolve the
// finest octave anyway and asking them to only adds noise.
#define LIGHT_MARCH_DETAIL 0.9

// Aerial perspective. The near turrets are 7 km away and the far ones 14, and
// seven kilometres of air is not nothing: this is the sea-level scattering
// coefficient scaled for the fact that much of the path is above the boundary
// layer, plus a flat aerosol term. Without it a cloud made of one material at
// one albedo has no depth cue at all, and the whole crown flattens into a
// decal. It is deliberately weak -- the subject is close and the reference is
// crisp -- but it is what separates the back of the crown from the front.
const vec3 BETA_AERIAL = BETA_R * 0.50 + vec3(1.2e-3);

// Optical depth from a point toward the sun. Six samples is not many, and where
// they land relative to a density boundary matters: on a fixed progression the
// crossings line up along whole regions of the surface and draw contour bands
// across the shading, which is the single most obvious artefact this pass can
// produce. Offsetting the progression by the same per-pixel, per-frame value
// the primary march uses turns those bands into noise, and noise is what
// Buffer B removes.
//
// Fewer octaves and no floret erosion: the fine scales get exponentiated and
// then smeared across the scattering orders, so they are not detail anyone
// can see -- but the turret scale is not optional, because one turret
// shadowing the next is most of what tells the eye it is looking at a heap
// of convective masses rather than at a painted gradient.
float lightMarch(sampler2D tex, vec3 p, vec3 sd, float time, float jitter) {
    float od = 0.0;
    float ds = LIGHT_STEP0;
    float dist = 0.0;
    for (int i = 0; i < LIGHT_STEPS; i++) {
        // One sample per segment, at a jittered position *within* the segment,
        // charged for the whole segment. The obvious cheaper jitter -- shift
        // the entire progression once by a fraction of the first step -- moves
        // every sample by 50 m at most, and the outer segments here are over a
        // kilometre long: a shadow edge crossing one of those still snaps the
        // optical depth by the whole segment at once, and the snap traces
        // perfect bullseye contours around every bump that casts a shadow.
        // In-segment jitter makes the estimate unbiased at any segment length,
        // so the contours dissolve into noise and the noise into the average.
        vec3 q = p + sd * (dist + jitter * ds);
        float sdf = stormSDF(q, time);
        if (sdf < INFLATE) {
            od += cloudDensity(tex, q, time, sdf, 2, LIGHT_MARCH_DETAIL) * ds;
        }
        dist += ds;
        ds *= LIGHT_GROWTH;
    }
    return od * EXTINCTION;
}

// Optical depth straight up. Three taps on the same geometric progression as
// the light march, reaching 1.4 km, which is far further in than skylight ever
// gets through a convective cloud.
#define AMBIENT_STEPS 3

float ambientMarch(sampler2D tex, vec3 p, float time, float jitter) {
    float od = 0.0;
    float ds = 0.12;
    float dist = 0.0;
    for (int i = 0; i < AMBIENT_STEPS; i++) {
        // In-segment jitter, for the same reason as lightMarch.
        vec3 q = p + vec3(0.0, dist + jitter * ds, 0.0);
        float sdf = stormSDF(q, time);
        if (sdf < INFLATE) {
            od += cloudDensity(tex, q, time, sdf, 1, 0.0) * ds;
        }
        dist += ds;
        ds *= 2.6;
    }
    return od * EXTINCTION;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec3 sd = sunDirection(iTime);

    // Sub-pixel jitter, resolved by Buffer B into an anti-aliased silhouette.
    vec2 ang = cameraAngles(iTime);
    vec3 ro, rd;
    cameraRay(fragCoord + haltonJitter(iFrame), iResolution, ang, ro, rd);

    vec3 scattered = vec3(0.0);
    float transmittance = 1.0;

    float t0, t1;
    if (boxRange(ro, rd, t0, t1)) {
        vec3 sunCol = SUN_RADIANCE * sunTransmittance(sd);
        vec3 skyLight = mix(skyRadiance(vec3(0.0, 1.0, 0.0), sd),
                            skyRadiance(normalize(vec3(sd.x, 0.22, sd.z)), sd),
                            0.15) * 3.8;
        vec3 haze = skyRadiance(rd, sd);
        float mu = dot(rd, sd);

        // Where the ray starts inside the box. Offsetting it by a per-pixel,
        // per-frame low-discrepancy value is what trades banding for noise --
        // and noise is what the temporal filter can remove.
        float offset = fract(ign(fragCoord) + float(iFrame) * 0.6180339887);
        float t = max(t0, 0.0);

        // Angular size of a pixel, which is what sets the step below. Tying the
        // step to the film rather than to a constant means the marcher tracks
        // the render resolution: a 96x64 check frame takes the coarse path and
        // costs a fraction of a 640x360 one, without a second set of tuned
        // numbers to keep in sync.
        float pixelAngle = 2.0 / (iResolution.y * FOCAL);

        float opticalDepth = 0.0;
        float skyDepth = 0.0;
        bool entered = false;

        for (int i = 0; i < MAX_STEPS; i++) {
            if (t > t1 || transmittance < MIN_TRANSMITTANCE) break;

            vec3 p = ro + rd * t;
            float sdf = stormSDF(p, iTime);

            // In-cloud step, held at a roughly constant angular size: the far
            // side of the anvil is 15 km further away than the near side of the
            // trunk and does not deserve the same sample density.
            //
            // Two pixels per step. That number is what the finest erosion
            // octave costs: at four the 200 m octave gets barely one sample per
            // wavelength and turns into vertical curtains along the view ray,
            // which the temporal filter then averages into a stable artefact
            // rather than removing. Halving the step doubles the pass and is
            // the price of the octave.
            float ds = clamp(t * pixelAngle * 2.0, 0.018, 0.30);

            if (sdf > INFLATE) {
                // Outside: jump. The SDF is a conservative bound on the density
                // field, so this skips space no sample would have found
                // anything in. 0.9 keeps it inside the Lipschitz bound the
                // smooth-min blends bend.
                t += max((sdf - INFLATE) * 0.9, ds);
                entered = false;
                continue;
            }

            // Just crossed in. Offsetting the first sample by a fraction of a
            // step is the whole reason the temporal filter has anything to
            // average: sphere tracing lands every ray on the surface itself, so
            // without this every pixel samples the same phase of the same
            // lattice and the result is a set of concentric shells, stable and
            // wrong. Re-done at every re-entry, because the jump above
            // resynchronises the lattice each time it fires.
            if (!entered) {
                entered = true;
                t += offset * ds;
                continue;
            }

            // Fade the fine erosion out as the step outgrows it. Detail smaller
            // than the step cannot be integrated, only aliased, and the
            // temporal filter will happily average aliasing into a stable,
            // permanent artefact.
            float detail = clamp(1.0 - (ds - 0.025) / 0.10, 0.0, 1.0);
            float density = cloudDensity(iChannel0, p, iTime, sdf, 3, detail);

            if (density > 0.001) {
                float sigma = density * EXTINCTION;

                if (density > LIGHT_MARCH_CUTOFF) {
                    opticalDepth = lightMarch(iChannel0, p, sd, iTime, offset);
                    skyDepth = ambientMarch(iChannel0, p, iTime, offset);
                }

                vec3 source = sunScatter(opticalDepth, mu, sunCol) +
                              ambientScatter(skyLight, skyDepth, opticalDepth);
                source *= ALBEDO;

                // Energy-conserving integration of the segment. Analytically
                // the radiance a segment adds is (S/sigma)(1 - exp(-sigma ds)),
                // and S is proportional to sigma, so the division cancels and
                // the whole thing collapses to one exp. Doing it this way
                // rather than S * sigma * ds is what makes the result
                // independent of the step length -- which matters a great deal
                // here, because the step length above changes by a factor of
                // four across the frame.
                float stepT = exp(-sigma * ds);
                vec3 segment = source * (1.0 - stepT);

                // Aerial perspective, per segment rather than per pixel,
                // because a cloud 11 km tall and 15 km deep has appreciably
                // more air in front of its far side than its near one.
                vec3 clear = exp(-BETA_AERIAL * t);
                segment = segment * clear + haze * (1.0 - stepT) * (1.0 - clear);

                scattered += transmittance * segment;
                transmittance *= stepT;
            }

            t += ds;
        }
    }

    fragColor = vec4(scattered, transmittance);
}
