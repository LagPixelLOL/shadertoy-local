// ============================================================================
//  Common -- the crown of a cumulonimbus, close up. One sun, one sky, nothing
//  else in frame.
//
//  What is being drawn is the top few kilometres of a convective cell that is
//  still growing: the stage before the anvil forms, when the updraft is a mass
//  of boiling turrets and the cloud is at its most three-dimensional. The rest
//  of the storm runs out of the bottom of the frame.
//
//  Units are kilometres and the scale is the real one. The crown is about
//  8 km across and its highest turret stands at 7 km; the camera is on the
//  ground 9.5 km away with a long lens, which is how the reference photographs
//  of this are taken and is what gives the compressed, stacked look. Keeping
//  the units physical is not pedantry: the extinction coefficient, the Rayleigh
//  cross-sections, the light-march reach and the turret sizes are all
//  quantities with published values, and inventing a unit means inventing all
//  of them again.
//
//  Nothing outside a #define or a comment names a pass uniform, so this file
//  also validates cleanly in shadertoy.com's Common tab -- see
//  examples/06-portable-common for why that matters.
// ============================================================================

const float PI = 3.14159265359;

// ---- daylight -------------------------------------------------------------
//
// The sun is the only light in the scene and its direction is fixed for now, so
// a frame is a function of the frame index alone and the temporal filter in
// Buffer B never has to reconverge on a lighting change.
//
// It is still plumbed as a variable everywhere downstream -- the sky, the light
// march, the phase function, the aerial perspective and the shafts all read it,
// none of them assume it is constant -- so making it dynamic is a change to
// this one function:
//
//     float az = SUN_AZIMUTH + 0.05 * time;
//     float el = SUN_ELEVATION + 0.30 * sin(0.09 * time);
//
// Every caller already passes iTime in. Two things will want attention when
// that happens: EXPOSURE has to follow the sun's elevation, because the scene
// spans several stops between noon and dusk, and Buffer B's blend weight has to
// rise or the history will smear the old lighting across the new.
//
// Low, to the left, and behind the camera's shoulder. That is the whole
// lighting design, and each of the three matters:
//
//   low       the beam has come through three and a half air masses, so it is
//             orange, and the sky it is not lighting is blue -- the
//             warm-against-cool split is what makes a lump of white vapour
//             read as a solid
//   lateral   a light on the camera's axis leaves no shadow and no shape; at
//             this angle every turret shades its own right-hand side
//   in front  the sun on the far side backlights the crown, which is a fine
//             picture but a different one: the near faces all go dark and the
//             detail the marcher is spending its budget on is not visible
//
// The azimuth wants to be very close to perpendicular to the view axis, and
// the window is narrow. sunDirection's z component is cos(az): at -2.00 it is
// -0.42, which puts the sun behind the camera, and a frontal light washes out
// the near faces into big flat lobes no matter how much relief the density
// field actually has -- the geometry is there, it just has no shading gradient
// to show it. At -1.30 the z component is +0.27 and the near faces are all in
// shadow. -1.60 is a hair past perpendicular, which keeps the shadow that
// gives the turrets their shape without losing the faces that carry the
// detail.
const float SUN_AZIMUTH   = -1.80;   // radians from +z, the direction of travel
const float SUN_ELEVATION =  0.26;   // radians above the horizontal

// Radiance of the solar disc before atmospheric extinction reddens it.
const vec3 SUN_RADIANCE = vec3(1.00, 0.98, 0.95) * 22.0;

vec3 sunDirection(float time) {
    float az = SUN_AZIMUTH;
    float el = SUN_ELEVATION;
    return vec3(sin(az) * cos(el), sin(el), cos(az) * cos(el));
}

// ---- the storm ------------------------------------------------------------

const float CLOUD_BASE = -2.5;   // below the frame; the cell continues down
const float CLOUD_TOP  =  7.4;

// Downwind direction of the shear that leans the column. Static: it tilts the
// shape, it does not move it.
const vec2 WIND_DIR = vec2(0.94391, 0.33037);   // normalize(vec2(1.0, 0.35))

// How fast the noise rises through the fixed shape, in km/s. Real updraft cores
// run 10-50 m/s; this is at the bottom of that, because a turret crossing its
// own diameter in twenty seconds is about as fast as the eye reads as boiling
// rather than as a conveyor belt.
const float BOIL_SPEED = 0.018;

// Axis-aligned bound on the shape below, with margin for INFLATE. The marcher
// clips to this before it does anything else, which is what makes the sky in
// the top corners cost one slab test.
const vec3 BOX_MIN = vec3(-10.5, CLOUD_BASE, -4.0);
const vec3 BOX_MAX = vec3( 10.5, 8.6,          6.5);

// Thickness of the rind the erosion noise works in, measured inward from the
// inflated surface. All the structure lives in this band: outside it there is
// no cloud, and inside it the density has already saturated, so there is
// nothing left to carve.
const float SHELL = 3.4;

// How far outside the analytic surface cloud is allowed to exist. The erosion
// only ever subtracts, and it subtracts more than a kilometre on average, so
// without this the rendered crown is a shrunken, rounded version of the shape
// in stormSDF. Offsetting the field outward and marching to the same offset
// restores the intended silhouette and keeps the SDF a conservative bound,
// which is the property the empty-space skipping depends on. Buffer A reads
// this: it is not free to disagree.
const float INFLATE = 1.2;

// Extinction of a convective cloud, per kilometre. Real cumulus sit around
// 20-60. At 24 a ray reaches 1% transmittance about 200 m into solid cloud,
// which is why the marcher terminates so quickly and why what is being drawn
// here is essentially a very rough surface.
const float EXTINCTION = 24.0;

// Single-scattering albedo. Cloud droplets are very nearly conservative
// scatterers -- the true albedo is 0.9999-something -- but water absorbs red
// more strongly than blue, so after the hundreds of scattering events a photon
// undergoes inside a cloud, what comes back out is measurably cool. This model
// runs three scattering orders, not hundreds, so the tilt is exaggerated to
// show at the order count we can afford what the real thing does.
const vec3 ALBEDO = vec3(0.955, 0.972, 0.995);

// ---- camera ---------------------------------------------------------------
//
// The position is a constant and only the orientation animates. That is not
// laziness, it is the precondition for Buffer B: reprojecting last frame into
// this one is exact for a pure rotation at every depth, because a rotation
// produces no parallax. Translate the camera and the reprojection needs a depth
// per pixel, which a volume does not have -- there is no single surface to
// reproject, and picking one smears exactly the soft edges this is meant to
// resolve.
const vec3  CAM_POS = vec3(0.0, 0.35, -10.5);

// A long lens: 54 degrees across the frame, 32 down it. Wide angle would put
// the crown in the middle of a lot of sky and flatten the stacking of one
// turret behind another, which is the read a reference photograph of this gets
// from being shot at 200 mm from ten kilometres away.
const float FOCAL = 3.5;

// Yaw and pitch. A slow, small drift: enough that the reprojection is doing
// real work rather than being the identity, small enough to keep the crown
// framed without having to move the camera. The amplitudes are small because
// the lens is long -- at this focal length a degree of yaw is a fortieth of the
// frame.
vec2 cameraAngles(float time) {
    return vec2(0.012 * sin(time * 0.051),
                0.385 + 0.005 * sin(time * 0.037 + 1.1));
}

void cameraRay(vec2 fragCoord, vec3 res, vec2 ang, out vec3 ro, out vec3 rd) {
    vec2 uv = (2.0 * fragCoord - res.xy) / res.y;
    vec3 v = normalize(vec3(uv, FOCAL));
    float cp = cos(ang.y), sp = sin(ang.y);
    v = vec3(v.x, cp * v.y + sp * v.z, cp * v.z - sp * v.y);   // pitch
    float cy = cos(ang.x), sy = sin(ang.x);
    v = vec3(cy * v.x + sy * v.z, v.y, cy * v.z - sy * v.x);   // yaw
    ro = CAM_POS;
    rd = normalize(v);
}

// Inverse of cameraRay: where a world direction lands on the film. Returns a
// large negative sentinel behind the camera; every caller range-checks the
// result against the viewport anyway.
vec2 cameraProject(vec3 rd, vec3 res, vec2 ang) {
    float cy = cos(ang.x), sy = sin(ang.x);
    vec3 v = vec3(cy * rd.x - sy * rd.z, rd.y, cy * rd.z + sy * rd.x);
    float cp = cos(ang.y), sp = sin(ang.y);
    v = vec3(v.x, cp * v.y - sp * v.z, cp * v.z + sp * v.y);
    if (v.z <= 1e-4) return vec2(-1e9);
    vec2 uv = v.xy * (FOCAL / v.z);
    return (uv * res.y + res.xy) * 0.5;
}

// ---- small utilities ------------------------------------------------------

float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

float smax(float a, float b, float k) {
    return -smin(-a, -b, k);
}

float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

// iq's round cone: the segment between two spheres. The natural primitive for a
// convective tower, which really is a stack of swelling bubbles.
float sdRoundCone(vec3 p, float r1, float r2, float h) {
    vec2 q = vec2(length(p.xz), p.y);
    float b = (r1 - r2) / h;
    float a = sqrt(max(1.0 - b * b, 0.0));
    float k = dot(q, vec2(-b, a));
    if (k < 0.0) return length(q) - r1;
    if (k > a * h) return length(q - vec2(0.0, h)) - r2;
    return dot(q, vec2(a, b)) - r1;
}

// iq's ellipsoid: an underestimate of the true distance, which is the safe
// direction to be wrong in for sphere tracing.
float sdEllipsoid(vec3 p, vec3 r) {
    float k0 = length(p / r);
    float k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / max(k1, 1e-6);
}

// Slab test against the crown's bounding box. Components of rd that are exactly
// zero are nudged rather than divided by: 0 * inf is NaN, and one NaN here
// silently deletes the cloud from a whole column of pixels.
bool boxRange(vec3 ro, vec3 rd, out float t0, out float t1) {
    vec3 d = rd + vec3(lessThan(abs(rd), vec3(1e-8))) * 1e-8;
    vec3 inv = 1.0 / d;
    vec3 a = (BOX_MIN - ro) * inv;
    vec3 b = (BOX_MAX - ro) * inv;
    vec3 lo = min(a, b), hi = max(a, b);
    t0 = max(max(lo.x, lo.y), lo.z);
    t1 = min(min(hi.x, hi.y), hi.z);
    return t1 > max(t0, 0.0);
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Interleaved gradient noise: a spatial dither with a near blue-noise spectrum
// for the cost of one dot product and one fract. Offsetting it by the golden
// ratio each frame makes the per-pixel sequence low-discrepancy in time as
// well, which is exactly what a temporal filter wants to be handed.
float ign(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

float radicalInverse(int i, int base) {
    float f = 1.0, r = 0.0;
    for (int k = 0; k < 16; k++) {
        if (i <= 0) break;
        f /= float(base);
        r += f * float(i - base * (i / base));
        i /= base;
    }
    return r;
}

// Halton(2,3): the sub-pixel jitter. 16 frames is the history length Buffer B's
// blend weight implies, so the sequence repeats on the same period the filter
// forgets, and no sample position is ever over-represented.
vec2 haltonJitter(int frame) {
    int i = frame - 16 * (frame / 16) + 1;
    return vec2(radicalInverse(i, 2), radicalInverse(i, 3)) - 0.5;
}

// ---- noise basis ----------------------------------------------------------
//
// Three-dimensional value noise out of a 256x256 tile of white noise. This is
// iq's trick -- lay the z slices across the tile with a per-slice offset and let
// the texture unit do the xy interpolation -- and it is what makes seven
// octaves affordable at hundreds of samples per pixel. An ALU hash needs eight
// evaluations per octave; this needs two texture reads that never leave L1.
//
// Two rather than one, deliberately. shadertoy.com's stock RGBA Noise Medium is
// built so that its green channel is its red channel shifted by (37,17), which
// lets the classic version read both z slices from a single fetch. Nothing
// requires a noise tile to have that property, and shadertoy-local's builtin,
// being independent white noise per channel, does not: relying on it would mix
// two unrelated slices and leave a discontinuity at every integer z. One extra
// L1 hit buys correctness against any tile, on the site as well as here.
const float NOISE_TILE = 256.0;

float vnoise(sampler2D tex, vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    vec2 uv = i.xy + vec2(37.0, 17.0) * i.z + f.xy + 0.5;
    // Two channels per fetch: .x is the value, .w -- independent white noise
    // in its own right -- serves as a low byte. The tile is 8-bit, and eight
    // bits is not enough for a field this slow: a two-cycle-per-kilometre
    // octave crosses one quantisation step every couple of hundred metres,
    // and behind a density ramp as steep as this shader's every step draws a
    // visible contour line across the crown. Since .x and .w arrive in the
    // same L1 hit, sixteen bits cost exactly nothing.
    vec4 ta = textureLod(tex, uv / NOISE_TILE, 0.0);
    vec4 tb = textureLod(tex, (uv + vec2(37.0, 17.0)) / NOISE_TILE, 0.0);
    float a = dot(ta.xw, vec2(1.0, 1.0 / 255.0)) * (255.0 / 256.0);
    float b = dot(tb.xw, vec2(1.0, 1.0 / 255.0)) * (255.0 / 256.0);
    return mix(a, b, f.z);
}

// The same lookup, returning all three channels instead of one. The tile's
// channels are independent white noise, so this is three independent noise
// fields -- a whole 3D vector field -- for the two reads a single scalar octave
// costs. Used for the domain warp below.
vec3 vnoise3(sampler2D tex, vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    vec2 uv = i.xy + vec2(37.0, 17.0) * i.z + f.xy + 0.5;
    vec3 a = textureLod(tex, uv / NOISE_TILE, 0.0).xyz;
    vec3 b = textureLod(tex, (uv + vec2(37.0, 17.0)) / NOISE_TILE, 0.0).xyz;
    return mix(a, b, f.z);
}

// Orthonormal, so successive octaves are decorrelated without changing scale.
const mat3 NOISE_ROT = mat3( 0.00,  0.80,  0.60,
                            -0.80,  0.36, -0.48,
                            -0.60, -0.48,  0.64);

// Fractal Brownian motion over the value noise, stretched to fill its range.
//
// The obvious choice for cauliflower is billow noise -- fold each octave
// through 1 - |2n-1| so the peaks round off and the valleys crease -- and it is
// wrong here, in a way worth recording because the picture it produces is
// unmistakable. That fold peaks wherever the underlying noise passes through
// 0.5, and in three dimensions the set of points where a smooth field equals a
// constant is a *surface*, not a set of points. Thresholding it therefore
// carves shells: the crown comes out looking like coral or bath foam, thousands
// of little rings, and no amount of retuning the amplitudes fixes it because
// the topology is wrong.
//
// Plain value noise has isolated local maxima. Thresholding it high leaves
// isolated convex blobs; lowering the threshold makes them swell and merge.
// That is exactly a cauliflower surface, and it is exactly what the erosion
// below does, because the threshold it compares against falls with depth into
// the rind.
//
// The smoothstep is not decoration. Interpolated white noise is nothing like
// uniform -- it piles up hard around its mean, and summing octaves narrows it
// further -- so the raw sum only ever varies over the middle fifth of its
// nominal range. Used directly as an erosion amount it moves the silhouette a
// couple of hundred metres and the crown stays a smooth dome. The two constants
// are the measured 5th and 95th percentiles of the four-octave sum.
float cloudFbm(sampler2D tex, vec3 p, int octaves) {
    float sum = 0.0, amp = 0.5, norm = 0.0;
    for (int i = 0; i < 5; i++) {
        if (i >= octaves) break;
        sum  += amp * vnoise(tex, p);
        norm += amp;
        p = NOISE_ROT * p * 2.03 + 0.31;
        amp *= 0.55;
    }
    return smoothstep(0.24, 0.80, sum / norm);
}

// The same fractal *without* the stretch, for the erosion. The stretch fills
// the range by pinning the distribution's tails, and a pinned tail is a
// *plateau*: a connected region where the field's gradient is exactly zero.
// Fed into the erosion, every such region carves a constant depth, so the
// surface there is the analytic envelope again -- offset, but exactly as
// smooth -- and the crown rendered as a stack of huge smooth facets with
// crinkle only along their joints. The erosion wants the opposite trade:
// span the range by *amplitude* (its scale factor is free) and keep a
// nonzero gradient everywhere.
float eroFbm(sampler2D tex, vec3 p, int octaves) {
    float sum = 0.0, amp = 0.5, norm = 0.0;
    for (int i = 0; i < 6; i++) {
        if (i >= octaves) break;
        sum  += amp * vnoise(tex, p);
        norm += amp;
        p = NOISE_ROT * p * 2.03 + 0.31;
        amp *= 0.55;
    }
    return sum / norm;
}

// ---- erosion frame --------------------------------------------------------
//
// An arbitrary second rotation, used together with NOISE_ROT to give each
// noise field its own oblique frame so their lattices never align with the
// world axes, the camera, or each other.
//
// (A full cellular/Worley basis lived here for a long time and is gone on
// purpose. `1 - F1` has one seductive property -- every maximum is an
// isolated point, so thresholding it gives convex bubbles -- and one fatal
// one: every one of those bubbles is a *circle*, the same circle, at every
// scale. Warping the domain, unioning two fields at incommensurate scales,
// mixing cell sizes: all of it was tried, and all of it produces warped,
// mixed-size circles. A cumulus crown is folded and kneaded, elongated as
// often as round, and that is the geometry of thresholded *smooth* noise,
// not of a distance field to a point set.)
const mat3 EROSION_ROT = mat3(0.00, -0.80, -0.60,
                              0.80,  0.36, -0.48,
                              0.60, -0.48,  0.64);

// Vertical compression of the erosion lookup domain: a turret is an updraft
// and stands taller than it is wide. This is deliberately mild -- 1/0.85 is
// about 1.2:1. Anything near the 1.7:1 the turrets actually measure turns the
// crown into vertical grooves, because the erosion stretches with it and the
// creases between bulges stretch furthest; the height wants to come from the
// analytic shape and the updraft lean, not from the noise basis.
const vec3 EROSION_ANISO = vec3(1.0, 0.85, 1.0);

// ---- shape ----------------------------------------------------------------
//
// Four primitives and no noise at all. This is only the big silhouette -- the
// stack of three crowns and the column they sit on. Every turret, every seam
// and every wisp comes from the erosion in cloudDensity.
//
// Keeping the shape analytic is what lets the marcher sphere-trace the empty
// space around it: the erosion only ever subtracts, and never outside the
// inflated surface, so this really is a conservative bound and a ray can jump
// the whole distance to it without sampling a single octave.
float stormSDF(vec3 p, float bob) {
    // Wind shear: the column leans downwind in proportion to how far it has
    // risen.
    float h = clamp((p.y - CLOUD_BASE) / (CLOUD_TOP - CLOUD_BASE), 0.0, 1.0);
    p.xz -= WIND_DIR * (1.5 * h * h);

    // The three crowns, at three heights. Real cells grow as a cluster of
    // updraft cores that overtake each other, so a single dome reads as a
    // balloon no matter what is carved into it: the tallest one has to be
    // off-centre and the others have to be visibly lower.
    //
    // Each one breathes on its own slow period. That is the second half of the
    // animation: the noise boiling upward gives the surface its motion, and
    // this gives the silhouette itself somewhere to go, so the crown grows and
    // subsides instead of being a fixed mould with weather running over it. The
    // amplitudes are a couple of hundred metres and none of them is horizontal,
    // so the cell stays exactly where it is.
    float b1 = 0.22 * sin(bob * 0.061);
    float b2 = 0.18 * sin(bob * 0.083 + 2.1);
    float b3 = 0.20 * sin(bob * 0.047 + 4.3);

    float d = sdEllipsoid(p - vec3( 0.2, 4.15 + b1, 1.0), vec3(3.6, 2.85, 2.6));
    d = smin(d, sdEllipsoid(p - vec3(-3.7, 3.10 + b2, 0.2), vec3(2.7, 2.10, 2.1)), 1.4);
    d = smin(d, sdEllipsoid(p - vec3( 3.6, 3.55 + b3, 1.5), vec3(2.8, 2.30, 2.2)), 1.4);

    // The mass below, running out of the bottom of the frame. Wide and shallow
    // rather than a column: a cylinder of the width this needs would put its
    // near face three kilometres from the camera, and the whole crown would
    // then be hidden behind a foreground bulge at a completely different scale.
    // Everything in the scene wants to sit at roughly one distance.
    d = smin(d, sdEllipsoid(p - vec3(0.0, 0.10, 1.2), vec3(8.6, 3.20, 2.9)), 1.8);
    return d;
}

// ---- density --------------------------------------------------------------
//
// *sdf* is stormSDF(p), taken as a parameter because every caller has already
// computed it to choose a step length and it is the second most expensive thing
// here. *detail* fades the high-frequency erosion out as the step grows: it is
// the level-of-detail knob, and sub-step features cannot be integrated, only
// aliased.
float cloudDensity(sampler2D tex, vec3 p, float time, float sdf,
                   int octaves, float detail) {
    // Depth into the rind, normalised. Zero at the inflated surface, one where
    // the shape is thick enough that no amount of erosion reaches.
    float base = clamp((INFLATE - sdf) / SHELL, 0.0, 1.0);
    if (base <= 0.0) return 0.0;

    // The only animation, and it is vertical only. A convective cell boils: the
    // turrets rise, overturn and are replaced, while the cell itself stays put.
    // Advecting the noise horizontally as well is the obvious way to add life
    // and is wrong here -- it slides every feature sideways, which reads as the
    // whole cloud drifting out of frame even though the analytic shape has not
    // moved at all.
    vec3 q = p;
    q.y -= time * BOIL_SPEED;

    // The warp that everything below reads through. This is what replaced the
    // cellular basis: the erosion is value fBm on a *folded* domain. Plain
    // fBm is too placid for convection and cellular noise is worse -- every
    // level set of 1 - F1 is a sphere, so every bump at every scale is a
    // circle, and no warp, union or size mixing ever stopped the render
    // reading as a pile of balls. Folding the domain is what a real updraft
    // does to a passive tracer. The warp's wavelength sits just under the
    // turret spacing and its amplitude well under a wavelength: bigger and
    // slower warps shear the field's level sets into long parallel streaks
    // -- the render grows sedimentary layers (tried, twice).
    vec3 qw = q + (vnoise3(tex, q * 0.90) - 0.5) * 0.55;

    // Large scale first: which regions of the crown bulge and which are
    // recessed. It is not making turrets, it is making the turrets uneven.
    float carve = (1.0 - cloudFbm(tex, q * 0.30, octaves)) * 0.14;

    // The erosion, in two parts, and the split is the load-bearing decision.
    // What a marcher renders is not an isosurface, it is the envelope of
    // whatever it integrates through: erode with one statistically uniform
    // field -- any field, any amplitude -- and a grazing ray always hits the
    // next lump within a wavelength or two, and the crown draws as the
    // smooth analytic envelope however violent the field is (tried, at
    // three amplitudes: dome, dome, flying rags -- the middle ground does
    // not exist). A silhouette is only ever made of *discrete, sparse*
    // masses. So:
    //
    //   turrets   the coarse fBm pushed through a hard threshold, so that
    //             only its isolated upper reaches survive as bulges a
    //             kilometre wide sticking out of the shaved-down skin.
    //             Blobs of a smooth random field: irregular in size, soft,
    //             elongated as often as round -- everything the Worley
    //             version's identical circles were not.
    //
    //   florets   signed fBm laid over everything, full strength at the
    //             skin and tapering to a third by mid-rind, so the skin is
    //             crinkled but the core stays solid instead of going porous.
    float tf = eroFbm(tex, qw * EROSION_ANISO * 0.95, 3);
    float turret = smoothstep(0.38, 0.72, tf);

    float fine = eroFbm(tex, NOISE_ROT * (qw * mix(EROSION_ANISO, vec3(1.0), 0.6))
                                 * 2.4 + 9.7, octaves + 1) - 0.5;

    carve += 0.34 - 0.28 * turret
                  - fine * 0.24 * (1.0 - 0.65 * smoothstep(0.10, 0.80, base));

    // A bulge may erode the carve to nothing but never below it: negative
    // carve would be cloud outside the inflated envelope, and the marcher's
    // empty-space jumps assume there is no such thing. Soft, because a hard
    // clamp is a plateau and a plateau renders as an analytic facet.
    carve = 0.02 + smax(carve - 0.02, 0.0, 0.08);

    // Depth below the *carved* surface, in rind units: zero at the visible
    // skin wherever the skin ends up. Gates the dither below so it does not
    // pay for itself deep inside the cloud where nothing can be seen.
    float surf = base - carve;
    float near = 1.0 - smoothstep(0.02, 0.34, surf);
    float gate = near * detail;

    // The florets themselves: a third, finer fractal that only the skin pays
    // for, gated on distance below the carved surface so the solid interior
    // never evaluates it. This one is *inverted billow* -- peaked where the
    // field crosses its mid level, zero at the extremes -- because at these
    // wavelengths (300 m and down) what the reference shows is not isolated
    // bumps but a *contiguous* tiling of small lobes with shallow seams
    // between them, and a signed field cannot tile: it is near zero over
    // most of its volume, so it decorates the skin in patches and leaves
    // airbrushed blanks between them (tried -- the faces read as dough with
    // occasional warts). The mid-level set of a smooth field is one
    // connected surface, so its crossings seam the whole skin without gaps.
    // The same fold at *coarse* scale renders as dried mud; shallow and
    // fine, it is exactly the brain-fold shading of a real crown. The last
    // factor fades it where the rind is thin, because a seam that carves
    // through a thin skyline leaves confetti hanging in the sky.
    if (gate > 0.02) {
        float n = eroFbm(tex, EROSION_ROT * qw * 6.0 + 3.1, octaves);
        float b = 1.0 - abs(2.0 * n - 1.0);
        carve += (b * b - 0.38) * 0.10 * gate * smoothstep(0.02, 0.12, base);
    }

    // The finest octave is *not* gated on `near`, and that is load-bearing: it
    // is also the dither that hides the tile's quantisation. The noise texture
    // is 8-bit, so the slow three-kilometre fBm crosses its 256 levels in
    // visible steps, and behind a density ramp this steep each step draws a
    // contour line -- nested onion rings, worst in exactly the deep creases
    // where a depth-gated octave has already faded out. 77 m of jitter at an
    // amplitude above one quantisation step breaks the contours everywhere the
    // surface can be seen.
    carve += (1.0 - cloudFbm(tex, q * 13.0 - 27.3, 2)) *
             max(gate, 0.35) * 0.026;

    // Cap how deep the carving reaches. The contributions above sum to about
    // 0.75 where their creases align, and a crease that deep is a canyon: it
    // cuts through the whole rind wherever the shape is under two kilometres
    // thick, and the sky shows through the middle of the crown in vertical
    // slots. The reference has creases everywhere and holes nowhere -- a
    // cumulus crease is shadowed vapour, not a window. A smooth minimum keeps
    // everything shallower than the knee exactly as it was, so the surface and
    // its spectrum do not move; it only stops the deepest crease intersections
    // from reaching the interior, which is what turns the crown from a cluster
    // of separate puffs back into one connected mass.
    carve = smin(carve, 0.62, 0.12);

    // Three numbers, and all three matter:
    //
    //   SHELL * carve   how far the surface moves -- the amplitudes sum to a
    //                   little over 0.8, so up to about 2.8 km, and the rind
    //                   has to be deep enough to hold that. A thin
    //                   rind lets the SDF's own gradient dominate the position
    //                   of the surface, and then the noise only ever shaves an
    //                   even skin off an analytic dome.
    //   SHELL / K       how thick the soft edge behind it is -- here 31 m,
    //                   which is about what a real cumulus edge measures. This
    //                   is also a low-pass filter on everything above: an
    //                   octave that displaces the surface by less than the edge
    //                   is thick cannot be seen at all. At 210 m -- the value
    //                   this had while the finest octaves were being tuned --
    //                   the top octaves were invisible and the crown rendered as
    //                   melted wax. Sharp edges and fine detail are not
    //                   alternatives, they are the same setting. 49 m was enough
    //                   for the old two-tier field but clipped the turbulence's
    //                   top octave; going to 31 m is worth about a tenth of the
    //                   power above 128 cycles and costs nothing in temporal
    //                   stability, which was the thing to check -- a sharper
    //                   edge is more high-frequency for the temporal filter to
    //                   swallow, and here it swallowed it (frame-to-frame mean
    //                   absolute difference 0.00301 against 0.00306).
    //   (1 - maxCarve) * K   must stay above 1, or the deep interior grows
    //                   holes and the crown shreds.
    float dens = clamp((base - carve) * 150.0, 0.0, 1.0);

    // Fade the outermost hundred metres of the rind. Where the rind is thin --
    // the skyline -- a coarse-cell bulge can survive the carving as an island,
    // and an island renders as a hard-edged dot hanging in the sky beside the
    // crown. Real fragments exist, but they are torn wisps, translucent and
    // fading; making anything that lives entirely in the outer skin optically
    // thin turns the dots into faint shreds and puts a soft fringe on the
    // silhouette, which the reference has.
    dens *= smoothstep(0.0, 0.10, base);

    // Thin the very tops, which are the youngest and least developed part of
    // the cell and are where it shears off into wisps.
    float h = clamp((p.y - CLOUD_BASE) / (CLOUD_TOP - CLOUD_BASE), 0.0, 1.0);
    return dens * (1.0 - 0.30 * smoothstep(0.82, 1.0, h));
}

// ---- atmosphere -----------------------------------------------------------
//
// Single-scattering Rayleigh plus Mie, integrated analytically by assuming the
// source term is constant along the view ray. That assumption is wrong and the
// error shows as a horizon slightly too dark at low sun; what it buys is a sky
// that costs a dozen instructions instead of a nested march, which matters
// because the cloud pass evaluates it once per pixel for aerial perspective and
// the image pass evaluates it again for the background.
//
// Coefficients are the standard sea-level values, per kilometre.
const vec3  BETA_R = vec3(5.802e-3, 13.558e-3, 33.100e-3);
const float BETA_M = 3.996e-3;
const float H_R = 8.0;    // Rayleigh scale height, km
const float H_M = 1.2;    // Mie scale height, km

// Ozone. It scatters nothing at all -- it is pure absorption, in the Chappuis
// band, which sits on green and red and leaves blue almost untouched. Leaving
// it out is why a single-scattering sky turns grey-brown as the sun drops:
// every path is being attenuated by Rayleigh alone, and Rayleigh attenuation
// at a low sun reddens the source faster than it blues the sky, so the whole
// dome desaturates. Measured against the reference crop the sky wanted a
// linear blue-to-red ratio of about 5.5 and the model was giving 2.3, falling
// to 1.6 once the sun was low enough to make the cloud the right colour.
//
// The layer is not a scale height -- ozone sits in a band around 25 km rather
// than falling off exponentially -- so this is its equivalent thickness, and
// it enters the extinction only, never the scattering source below.
const vec3  BETA_O = vec3(0.650e-3, 1.881e-3, 0.085e-3);
const float H_O = 15.0;   // equivalent ozone layer thickness, km

// Strength of the multiple-scattering term in skyRadiance. See there.
const float MS_STRENGTH = 0.0140;

// Relative air mass. Exact at the zenith, about 38 at the horizon, which is the
// right ballpark, and unlike sec(z) it does not diverge.
float airMass(float cosZenith) {
    return 38.0 / (37.0 * clamp(cosZenith, 0.0, 1.0) + 1.0);
}

// Vertical column optical depth at the zenith. Extinction, so ozone is in it;
// the scattering source term below divides by this, which is exactly the
// single-scattering albedo doing its job -- absorbed light is light that never
// reaches the camera from any direction.
const vec3 ZENITH_OD = BETA_R * H_R + vec3(BETA_M * H_M) + BETA_O * H_O;

vec3 sunTransmittance(vec3 sd) {
    return exp(-ZENITH_OD * airMass(sd.y));
}

vec3 skyRadiance(vec3 rd, vec3 sd) {
    float mu = clamp(dot(rd, sd), -1.0, 1.0);
    float phaseR = 3.0 / (16.0 * PI) * (1.0 + mu * mu);
    float g = 0.76, g2 = 0.5776;
    float phaseM = (1.0 - g2) /
                   (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * mu, 1e-4), 1.5));

    float m = airMass(rd.y);
    vec3 view = 1.0 - exp(-ZENITH_OD * m);

    // Scattering-coefficient-weighted phase, divided back out by the extinction
    // the same air mass produces: the air mass cancels, leaving the fraction of
    // the beam that ends up coming at the camera.
    vec3 j = (BETA_R * H_R * phaseR + vec3(BETA_M * H_M * phaseM)) / ZENITH_OD;

    vec3 col = SUN_RADIANCE * sunTransmittance(sd) * j * view;

    // Second order. Single scattering alone cannot
    // keep a low sun's sky blue and it is worth being precise about why: the
    // scattering itself is fine -- at this view angle the blue-to-red ratio of
    // (phase * (1 - exp(-od))) is about 4.8, which is a proper sky -- but the
    // whole dome is then multiplied by sunTransmittance, and a beam that has
    // come through three air masses carries a blue-to-red ratio of 0.53. The
    // product is 2.4, and the sky renders grey-brown. Lowering the sun to get
    // the warm light the reference has makes it worse, not better: at an
    // elevation of 0.19 it falls to 1.6.
    //
    // The light that keeps a real sunset sky blue overhead has bounced more
    // than once, and it did not come along the horizon path -- it was scattered
    // high and sideways, so it is nothing like as reddened, and having been
    // Rayleigh-scattered at every bounce it is far bluer than the beam that
    // produced it. Hence both changes here: the tint is the Rayleigh
    // coefficients themselves rather than the warm grey this used to be, and
    // the transmittance is evaluated at a floored elevation so the term does
    // not redden in lockstep with the direct beam it is supposed to be
    // compensating for.
    //
    // Both are the right shape and neither is a fix: measured, this moves the
    // sky's blue-to-red ratio from 2.31 to 2.48 across a fourfold sweep of
    // MS_STRENGTH, because the term is only about a seventh of the blue and
    // its own tint still carries some red. Reaching the reference's 5.5 this
    // way needs roughly thirty times the energy, which is a white sky. The
    // rest of the gap is not a scattering problem and is dealt with by the
    // grade in tonemap().
    vec3 msTint = BETA_R / BETA_R.b;
    vec3 msT = exp(-ZENITH_OD * airMass(max(sd.y, 0.35)));
    col += SUN_RADIANCE * MS_STRENGTH * msTint * msT * view * max(sd.y, 0.0);
    return col;
}

// The disc itself, at roughly its real angular radius of a quarter of a degree,
// plus two glow lobes: the tight one is forward-scattered sunlight in the air
// around it, the wide one is what any lens does with a source this bright. The
// sun is well outside this frame, but the wide lobe is not, and it is what
// lifts and warms the sky on the sunward side.
vec3 sunDisc(vec3 rd, vec3 sd) {
    float mu = dot(rd, sd);
    vec3 t = SUN_RADIANCE * sunTransmittance(sd);
    vec3 col = t * 240.0 * smoothstep(0.99988, 0.99996, mu);
    col += t * 0.85 * pow(max(mu, 0.0), 2200.0);
    col += t * 0.020 * pow(max(mu, 0.0), 130.0);
    return col;
}

// ---- scattering in the cloud ----------------------------------------------

float hg(float g, float mu) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * mu, 1e-4), 1.5));
}

// Mie scattering off cloud droplets is overwhelmingly forward, but the small
// backward lobe is not optional: it is the glory, and the reason a cloud with
// the sun behind the viewer is bright rather than flat. One Henyey-Greenstein
// lobe cannot be both.
float cloudPhase(float mu, float ecc) {
    return mix(hg(0.80 * ecc, mu), hg(-0.35 * ecc, mu), 0.30);
}

// Multiple-scattering octaves, after Wrenninge et al. A single-scattering cloud
// is black wherever the sun does not reach directly, which is wrong by orders
// of magnitude: almost all the light leaving a cloud has bounced many times.
// Tracking that honestly is a path trace. This sums a few terms with
// progressively weaker extinction, weaker contribution and a flatter phase
// function, which is what deeper scattering orders behave like -- the light
// spreads, forgets which way it came from, and reaches further in than the
// direct beam ever does. It is also what fills the shadowed side of a turret
// with a soft glow instead of a hard terminator.
//
// The point is that all three terms reuse the *same* optical depth. The light
// march that produced it is the expensive part, and it is paid for once.
// Scattering orders, and how fast they die away. The turbulence rework made
// the crown much more deeply eroded than it was, and a deeply eroded cloud is
// full of creases that single scattering renders as near-black slots -- the
// frame came out with a median well under the reference's. Real cumulus is not
// dark in its creases, because a droplet cloud has an albedo of about 0.9998
// and the light in a crease has bounced a dozen times before it leaves. Adding
// orders is the honest fix for that; raising the exposure is not, and the tone
// curve's shoulder eats it anyway.
#define MS_ORDERS 4
const float MS_FALLOFF = 0.93;

vec3 sunScatter(float opticalDepth, float mu, vec3 sunCol) {
    vec3 l = vec3(0.0);
    float a = 1.0, b = 1.0, c = 1.0;
    for (int i = 0; i < MS_ORDERS; i++) {
        l += b * exp(-a * opticalDepth) * cloudPhase(mu, c);
        a *= 0.40;          // extinction: deeper orders are attenuated less
        b *= MS_FALLOFF;    // contribution
        c *= 0.60;          // eccentricity: deeper orders are more isotropic
    }
    return l * sunCol;
}

// Skylight, which is the only thing lighting the shadowed side. *skyOD* is the
// optical depth straight up out of the cloud, which Buffer A measures with a
// three-tap march; the 0.5 is because ambient arrives from the whole hemisphere
// and one vertical ray overstates how much of it is blocked.
//
// Measuring it rather than approximating it is worth the three samples. The
// obvious cheap substitute -- attenuate by how far inside the analytic shape
// the sample is -- is a function of the SDF and nothing else, so it is
// perfectly smooth, and it makes every surface the sun does not reach a flat
// silhouette with no structure in it at all. On a cloud lit this laterally that
// is half the frame, and the seams between turrets are precisely where the eye
// is reading the shape.
vec3 ambientScatter(vec3 skyLight, float skyOD, float sunOD) {
    // Rational falloff, not exponential. Beer's law is the transmittance of an
    // unscattered beam, and skylight reaching a point a few hundred metres
    // inside a cloud has not been a beam for a long time -- it has diffused,
    // and diffusion falls off polynomially. Using exp() here drives the whole
    // shadowed side of the crown to black within about 300 m of the surface,
    // which is both wrong and the difference between a photograph and a
    // charcoal drawing: in the reference the shadows are the second brightest
    // thing in the frame, a luminous blue-grey, not an absence.
    // Deep in a crease the ambient is no longer sky-coloured either. What
    // reaches a point several hundred metres from open sky has mostly arrived
    // by bouncing off the white walls of the crease itself, and every bounce
    // off a droplet cloud is spectrally flat, so the deeper the point the
    // whiter its illumination. Without this the shadow pockets render in the
    // sky's own saturated blue and read as holes in the crown showing the sky
    // behind it -- the giveaway that they are lit by one skyRadiance sample
    // rather than by an environment that is mostly cloud.
    //
    // Two measures of "deep", because the vertical march alone cannot see the
    // case that matters most: a pocket on the shadowed flank of a turret is
    // open to the sky straight up, so its skyOD is small, but laterally it
    // faces the white wall of the next turret -- and laterally is where its
    // light comes from. The sun's optical depth is a serviceable proxy for
    // that enclosure: a point the direct beam cannot reach is a point sitting
    // behind cloud, and cloud is what it sees.
    float lum = dot(skyLight, vec3(0.2126, 0.7152, 0.0722));
    float desat = clamp(skyOD * 0.30 + sunOD * 0.10, 0.0, 0.70);
    vec3 amb = mix(skyLight, vec3(lum), desat);

    // ... and the falloff has a floor, for the same reason it is rational
    // rather than exponential, taken one step further. The reference's crown
    // never gets darker than about 0.65 -- not in the deepest seam it has.
    // Skylight attenuates going in, but every crease is also walled by
    // sunlit vapour, and vapour is white: past a few hundred metres the
    // radiance field inside the cloud stops falling because it is being fed
    // laterally from the walls. A pure falloff of any shape reaches zero
    // eventually and renders wells the photograph does not contain.
    return amb * (0.62 + 0.38 / (1.0 + skyOD * 0.36));
}

// ---- display --------------------------------------------------------------

// Exposure. A constant, because the sun is: see sunDirection above for what has
// to change here when it stops being one.
const float EXPOSURE = 0.42;

// Narkowicz's fit of the ACES filmic curve. A sunlit turret against a shadowed
// seam spans several stops, and a linear clamp throws the top of that away as a
// flat white blob -- which on a subject that is white to begin with is the
// whole picture.
vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
                 0.0, 1.0);
}

float acesL(float x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
                 0.0, 1.0);
}

// Applying that curve per channel is what a film emulation is supposed to do,
// and on this subject it is the single biggest thing standing between the
// render and the photograph. Each channel saturates on its own, so every pixel
// bright enough to matter is dragged toward white: measured against the
// reference crop, the lit faces came out at a median saturation of 0.08 where
// the photograph has 0.32, and no amount of exposure fixes it -- pulling the
// exposure down to 0.38 only reaches 0.15, and by then the midtone has fallen
// to 0.63 against the reference's 0.83. The curve trades chroma for highlight
// rolloff at exactly the brightness this image lives at.
//
// So run the curve on luminance and carry the chroma through unchanged. The
// warm-against-cool split that sunDirection's comment is about is entirely a
// chroma signal -- a golden lit face against a blue-grey shadow, both of them
// bright -- and it only survives if the tone curve leaves the ratios alone.
// Near white there is nowhere left to put the chroma, so fall back to the
// per-channel curve over the last stop, which is where its rolloff is worth
// having and where nothing is left to desaturate anyway.
// The grade, and it is a grade -- the reference is a JPEG out of a camera that
// applied a white balance and a saturation curve, and the last of the gap to it
// is that processing rather than any missing physics. Naming it here beats
// smuggling it into the scattering coefficients, which is the tempting move and
// leaves you with an atmosphere that is wrong everywhere else.
//
// It is a split tone: the shadows and the sky are lit by a blue hemisphere and
// the sunlit faces by a reddened beam, so tinting by luminance pushes the two
// apart along exactly the axis the subject already varies on. That is also why
// this is not a global saturation boost, which would only stretch the existing
// grey-green cast in both directions.
//
// The dark grade is *two* colours, selected by what the pixel is, because the
// two dark things in this frame are not the same colour and no luminance key
// can tell them apart: the sky sits in the same luma band as a shadowed crease.
// Measured in the reference, the sky is strongly blue (linear b/r 5.4) while
// the cloud's own shadows are very nearly neutral and a hair *warm* (mean
// b - r of -0.03 across the shadow band -- scattered sunlight has been
// reddened by three air masses, and it gets everywhere). One blue grade
// applied to both was what painted the creases sky-blue and made every shadow
// pocket read as a hole in the crown. The image pass knows the cloud's
// transmittance, which is exactly the selector needed, and passes it here.
const vec3 GRADE_SKY          = vec3(0.61, 1.16, 1.66);
const vec3 GRADE_CLOUD_SHADOW = vec3(1.21, 1.19, 1.09);
const vec3 GRADE_HIGHLIGHT    = vec3(1.20, 1.00, 0.74);

vec3 tonemap(vec3 x, float skyness) {
    float l = max(dot(x, vec3(0.2126, 0.7152, 0.0722)), 1e-5);
    float ln = acesL(l);
    vec3 ratio = x * (ln / l);
    vec3 c = mix(ratio, aces(x), smoothstep(0.85, 1.0, ln));
    vec3 dark = mix(GRADE_CLOUD_SHADOW, GRADE_SKY, skyness);
    c *= mix(dark, GRADE_HIGHLIGHT, smoothstep(0.0, 1.0, ln));
    return clamp(c, 0.0, 1.0);
}
