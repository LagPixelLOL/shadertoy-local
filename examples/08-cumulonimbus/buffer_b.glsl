// ============================================================================
//  Buffer B -- temporal resolve. Sixteen of Buffer A's one-sample frames,
//  reprojected and averaged into one.
//
//  iChannel0: Buffer A (rgb = scattered light, a = transmittance), nearest
//  iChannel1: Buffer B, i.e. this pass one frame ago, linear
//  output:    the same layout, filtered
//
//  This is the pass that pays for the detail. Buffer A takes one jittered
//  sample per pixel: on its own that is a grainy, aliased image with visible
//  stepping through the soft edges. Accumulating sixteen of them with a
//  different sub-pixel offset and a different march offset each time costs one
//  march per frame and converges on sixteen -- so the fine erosion octaves and
//  the coarse marching step are both affordable, and neither shows.
//
//  The reprojection is exact. cameraAngles rotates the camera and never moves
//  it, and a pure rotation produces no parallax at any depth, so last frame's
//  pixel for a given world direction can be found by inverting last frame's
//  orientation -- no depth buffer, and nothing that a volume would have to lie
//  about. This is the one reason the camera does not fly.
// ============================================================================

// One over the history length. 1/16 to match haltonJitter's period: shorter and
// the jitter sequence never completes, longer and the cloud's own motion starts
// to smear.
#define BLEND 0.0625

// Variance clipping width, in standard deviations of the 3x3 neighbourhood.
// The cloud evolves and the camera turns, so some of the history is always
// stale; clipping it into the range the current frame actually produced is what
// keeps that from becoming a trail. 1.25 is tight enough to kill visible
// ghosting on the anvil's leading edge and loose enough to still average.
#define CLIP_SIGMA 1.25

vec4 fetchA(ivec2 q) {
    return texelFetch(iChannel0, clamp(q, ivec2(0), ivec2(iResolution.xy) - 1), 0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    ivec2 q = ivec2(fragCoord);
    vec4 current = fetchA(q);

    // First and second moments of the 3x3 neighbourhood. Moments rather than a
    // min/max box: a single bright sample from a fine wisp would widen a box
    // enough to let a whole frame of ghost through, while it barely moves the
    // standard deviation.
    vec4 m1 = vec4(0.0), m2 = vec4(0.0);
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec4 s = fetchA(q + ivec2(x, y));
            m1 += s;
            m2 += s * s;
        }
    }
    m1 /= 9.0;
    m2 /= 9.0;
    vec4 sigma = sqrt(max(m2 - m1 * m1, vec4(0.0)));
    vec4 lo = m1 - CLIP_SIGMA * sigma;
    vec4 hi = m1 + CLIP_SIGMA * sigma;

    vec4 resolved = current;

    if (iFrame > 0) {
        vec2 ang = cameraAngles(iTime);
        vec2 prevAng = cameraAngles(iTime - iTimeDelta);
        vec3 ro, rd;
        cameraRay(fragCoord, iResolution, ang, ro, rd);
        vec2 prevCoord = cameraProject(rd, iResolution, prevAng);

        // Half a pixel in from the edge, so the bilinear tap cannot reach
        // outside the frame and clamp a wrong sample into the history.
        if (all(greaterThan(prevCoord, vec2(0.5))) &&
            all(lessThan(prevCoord, iResolution.xy - 0.5))) {
            vec4 history = texture(iChannel1, prevCoord / iResolution.xy);
            resolved = mix(clamp(history, lo, hi), current, BLEND);
        }
    }

    fragColor = resolved;
}
