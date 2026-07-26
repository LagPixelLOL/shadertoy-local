// ============================================================================
//  Buffer B -- temporal resolve. Sixteen of Buffer A's one-sample frames,
//  reprojected and averaged into one.
//
//  iChannel0: Buffer A (rgb = scattered light, a = transmittance), nearest
//  iChannel1: Buffer B, i.e. this pass one frame ago, linear (resampled
//             through a Catmull-Rom kernel; see historyCatmullRom)
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

// Catmull-Rom history resampling, in nine bilinear taps. The obvious single
// bilinear tap is not an option, and the reason is quantitative: the camera
// rotates a little every frame, so the reprojected position is always
// fractional, and the history is resampled *again every frame* -- sixteen
// bilinear resamples compound into a blur kernel a couple of pixels wide,
// which is exactly the scale of the floret detail the whole pipeline exists
// to resolve. Measured against an externally averaged stack of Buffer A
// frames, the bilinear version was visibly softer everywhere and the fine
// silhouette crenellation was simply gone. Catmull-Rom's negative lobes
// undo most of the successive-resample spread; what little over/undershoot
// they add is caught by the variance clip below, which was already there.
vec4 historyCatmullRom(vec2 pos, vec2 res) {
    vec2 centre = floor(pos - 0.5) + 0.5;
    vec2 f = pos - centre;
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    vec2 p0 = (centre - 1.0) / res;
    vec2 p3 = (centre + 2.0) / res;
    vec2 p12 = (centre + w2 / w12) / res;
    return
        texture(iChannel1, vec2(p0.x,  p0.y))  * (w0.x  * w0.y) +
        texture(iChannel1, vec2(p12.x, p0.y))  * (w12.x * w0.y) +
        texture(iChannel1, vec2(p3.x,  p0.y))  * (w3.x  * w0.y) +
        texture(iChannel1, vec2(p0.x,  p12.y)) * (w0.x  * w12.y) +
        texture(iChannel1, vec2(p12.x, p12.y)) * (w12.x * w12.y) +
        texture(iChannel1, vec2(p3.x,  p12.y)) * (w3.x  * w12.y) +
        texture(iChannel1, vec2(p0.x,  p3.y))  * (w0.x  * w3.y) +
        texture(iChannel1, vec2(p12.x, p3.y))  * (w12.x * w3.y) +
        texture(iChannel1, vec2(p3.x,  p3.y))  * (w3.x  * w3.y);
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

        // Two and a half pixels in from the edge, so no Catmull-Rom tap can
        // reach outside the frame and clamp a wrong sample into the history.
        if (all(greaterThan(prevCoord, vec2(2.5))) &&
            all(lessThan(prevCoord, iResolution.xy - 2.5))) {
            vec4 history = historyCatmullRom(prevCoord, iResolution.xy);
            resolved = mix(clamp(history, lo, hi), current, BLEND);
        }
    }

    fragColor = resolved;
}
