// ============================================================================
//  PATH-TRACED BOX -- Common tab
//
//  One closed room, lit by a frosted globe -- a scattering medium with a bright
//  filament inside, like an opal bulb -- that cycles hue and pulses on a 120 BPM
//  kick, plus a dim panel recessed into the ceiling for fill. The classic set
//  piece for showing what a path tracer buys you: soft shadows, colour bleeding
//  off four tinted walls and two checkered planes, a glossy floor, a mirror ball,
//  a tinted glass ball that refracts the room and spills a caustic onto the floor
//  (a soft one -- the dominant source here is large), and a volumetric source
//  that is genuinely marched rather than faked.
//
//  The renderer is split across five passes:
//    Buffer A  2 samples/pixel/frame + next event estimation, and a packed
//              G-buffer for the denoiser
//    Buffer B  temporal accumulation with reprojection and variance clipping
//    Buffer C  edge-aware a-trous wavelet, step 1
//    Buffer D  edge-aware a-trous wavelet, step 2
//    Image     edge-aware a-trous wavelet, step 4, then remodulate + tonemap
//
//  Shared here: RNG, scene, materials, BRDF, camera, G-buffer packing, tonemap.
//
//  Drag the mouse to aim the camera; it orbits on its own when you let go.
//
//  Every helper takes what it needs as a PARAMETER instead of reading a uniform,
//  so this file also validates cleanly in shadertoy.com's Common tab (the site
//  checks Common standalone, where iTime and friends do not exist).
// ============================================================================

#define PI  3.14159265358979
#define TAU 6.28318530717959

// Far plane. The room is ~8 m across the diagonal; a primary ray folded through
// three specular interfaces stays well under this.
const float FAR = 40.0;

// ============================================================================
//  interactive camera
// ============================================================================
//  Drag anywhere over the shader to aim: horizontal position picks the azimuth
//  (a full turn across the width), vertical position picks the elevation. With
//  the button up the camera keeps orbiting slowly at whatever elevation it was
//  left at, so the demo is never frozen.
//
//  The angles have to persist between frames, because the denoiser needs the
//  camera the PREVIOUS frame used in order to reproject its history, and a
//  mouse-driven camera cannot be recomputed from iTime. Buffer A therefore
//  integrates them into its (0,0) state pixel (it reads its own previous frame
//  on iChannel0) and publishes both angles there; every other pass fetches them
//  from Buffer A. Shadertoy only reports the mouse while a button is held
//  (iMouse.z > 0) - hover alone is not visible to shaders.
const float ORBIT_R   = 3.85;               // orbit radius (m)
const float ORBIT_Y   = 1.90;               // orbit centre height (m)
const vec3  LOOK_AT   = vec3(0.0, 1.05, 0.0);
const float FOCAL     = 1.25;               // ~44 deg vertical field of view
const vec2  ANG0      = vec2(1.20, 0.20);   // opening azimuth / elevation (rad)
// The elevation limits keep the camera inside the room and well clear of the
// props at every azimuth, so no drag can put the eye inside a wall or a ball.
const float PITCH_LO  = -0.16;              // elevation at the bottom of a drag
const float PITCH_HI  = 0.40;               // ... and at the top
const float IDLE_SPIN = 0.10;               // rad/s while the button is up
const float AIM_TAU   = 0.18;               // mouse follow time constant (s)

// (0,0) of Buffer A: xy = the angles THIS frame used, zw = the previous frame's.
vec4 camState(sampler2D aCh) { return texelFetch(aCh, ivec2(0, 0), 0); }

// Advance the orbit by one frame; returns vec4(current, previous) to be stored.
vec4 stepCam(vec4 state, vec4 mouse, vec3 res, float dt, int frame) {
    // dot(state, state) >= 0.0 is false for NaN, so one test covers a fresh
    // (cleared) buffer, a resize and a poisoned state pixel alike.
    float mag = dot(state, state);
    if (frame <= 0 || !(mag >= 0.0) || mag > 1e12) return vec4(ANG0, ANG0);

    vec2 ang = state.xy;
    if (mouse.z <= 0.0) return vec4(ang.x + IDLE_SPIN * dt, ang.y, ang);

    vec2 m = mouse.xy / res.xy;
    vec2 tgt = vec2(ANG0.x + (m.x - 0.5) * TAU,
                    mix(PITCH_LO, PITCH_HI, clamp(m.y, 0.0, 1.0)));
    // Unwrap the azimuth onto the nearest equivalent turn: the idle orbit winds
    // it up without bound, and grabbing the mouse must not unwind all of it.
    tgt.x += TAU * floor((ang.x - tgt.x) / TAU + 0.5);
    // Exponential follow rather than a jump: bounded per-frame motion is what
    // keeps Buffer B's reprojection useful while the camera is being dragged.
    return vec4(mix(ang, tgt, 1.0 - exp(-dt / AIM_TAU)), ang);
}

void getCam(vec2 ang, out vec3 ro, out mat3 cb, out float fl) {
    float cp = cos(ang.y);
    ro = vec3(cos(ang.x) * ORBIT_R * cp,
              ORBIT_Y + ORBIT_R * sin(ang.y),
              sin(ang.x) * ORBIT_R * cp);
    fl = FOCAL;
    vec3 fwd = normalize(LOOK_AT - ro);
    vec3 rt  = normalize(cross(fwd, vec3(0, 1, 0)));
    vec3 up  = cross(rt, fwd);
    cb = mat3(rt, up, fwd);
}
void camRay(vec2 fragCoord, vec3 res, vec2 ang, out vec3 ro, out vec3 rd) {
    mat3 cb; float fl;
    getCam(ang, ro, cb, fl);
    vec2 uvn = (fragCoord - 0.5 * res.xy) / res.y;
    rd = normalize(cb * vec3(uvn, fl));
}

// ============================================================================
//  RNG (pcg4d) & hashing
// ============================================================================
uvec4 _rs;
uvec4 pcg4d(uvec4 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    v ^= v >> 16u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    return v;
}
void seedRNG(vec2 fc, int frame) {
    _rs = uvec4(uint(fc.x), uint(fc.y), uint(frame), 0x9E3779B9u);
}
float rnd() {
    _rs = pcg4d(_rs);
    return float(_rs.x) * (1.0 / 4294967296.0);
}
vec2 rnd2() { _rs = pcg4d(_rs); return vec2(_rs.xy) * (1.0 / 4294967296.0); }

float hash12(vec2 p) {
    uvec2 q = floatBitsToUint(p);
    uvec4 h = pcg4d(uvec4(q, 0x2545F491u, 0x9E3779B9u));
    return float(h.x) * (1.0 / 4294967296.0);
}

// ============================================================================
//  small utilities
// ============================================================================
float lum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

mat3 orthoBasis(vec3 n) {
    vec3 t = abs(n.y) < 0.99 ? normalize(cross(n, vec3(0, 1, 0))) : vec3(1, 0, 0);
    vec3 b = cross(n, t);
    return mat3(t, b, n);
}
vec3 cosineDir(vec3 n, vec2 u) {
    float r = sqrt(u.x), ph = TAU * u.y;
    return orthoBasis(n) * vec3(r * cos(ph), r * sin(ph), sqrt(1.0 - u.x));
}
// Uniform on the sphere: the isotropic phase function, sampled exactly.
vec3 uniformSphereDir(vec2 u) {
    float z = 1.0 - 2.0 * u.x;
    float r = sqrt(max(1.0 - z * z, 0.0));
    float ph = TAU * u.y;
    return vec3(r * cos(ph), r * sin(ph), z);
}
// Uniform inside the cone of half-angle acos(cosAng) about d, so the pdf is the
// constant 1/(2*pi*(1 - cosAng)). Taking the cosine instead of the angle keeps
// an acos out of the inner loop.
vec3 sampleCone(vec3 d, float cosAng, vec2 u) {
    float ca = mix(cosAng, 1.0, u.x);
    float sa = sqrt(max(1.0 - ca * ca, 0.0));
    float ph = TAU * u.y;
    return orthoBasis(d) * vec3(cos(ph) * sa, sin(ph) * sa, ca);
}

// ============================================================================
//  scene
// ============================================================================
//  Room interior: |x| < ROOM.x, 0 < y < ROOM.y, |z| < ROOM.z. Everything the
//  camera can reach is inside it, which makes the wall test a single slab exit
//  (see roomHit) and means no ray ever escapes to an environment.
const vec3  ROOM     = vec3(4.20, 3.60, 4.20);
const vec2  LIGHT_R  = vec2(1.45, 1.10);      // ceiling panel half extents
const vec3  LIGHT_E  = vec3(1.0, 0.95, 0.88) * 2.4;    // emitted radiance
const float LIGHT_A  = 4.0 * LIGHT_R.x * LIGHT_R.y;    // panel area (m^2)

// The glass ball stands on a slim plinth rather than on the floor, and that is
// optics, not decoration. A solid sphere of this radius focuses about 0.18 m
// behind its own back surface, so resting on the floor it would do nothing but
// magnify the single tile it sits on (verified: the whole ball resolved to one
// tile), and its caustic would converge inside the contact point where nothing
// can see it. Lifted clear, it images the whole room instead, and the panel's
// light focuses past it onto open floor - which is the caustic below it.
const vec3  GLASS_C  = vec3(-1.06, 1.64,  0.19);
const float GLASS_R  = 0.54;
const vec3  PLINTH_C = vec3(-1.06, 0.55,  0.19);
const vec3  PLINTH_H = vec3( 0.18, 0.55,  0.18);
const vec3  METAL_C  = vec3( 0.98, 0.46,  0.78);
const float METAL_R  = 0.46;
const vec3  BOX_C    = vec3( 0.35, 0.57, -1.08);
const vec3  BOX_H    = vec3( 0.43, 0.57,  0.43);
const float BOX_ROT  = 0.42;                  // yaw of the block (rad)

const float IOR_GLASS = 1.52;
// Beer-Lambert absorption inside the ball (1/m). Faint, but it is what keeps
// the thick middle of the sphere from looking like vacuum.
const vec3  GLASS_ABS = vec3(0.30, 0.10, 0.16);
const vec3  METAL_F0  = vec3(0.95, 0.92, 0.86);

// Material ids are contiguous from 0 so the G-buffer packs them as id + 1,
// leaving 0 for "nothing hit". Keep them that way if you add one.
// All four side walls are tinted, not just two: the camera makes a full turn,
// so a white wall would mean half the orbit shows no colour bleeding at all.
#define MAT_NONE  -1
#define MAT_FLOOR  0
#define MAT_CEIL   1
#define MAT_RED     2   // -x
#define MAT_GREEN   3   // +x
#define MAT_BLUE    4   // -z
#define MAT_AMBER   5   // +z
#define MAT_BOX    6
#define MAT_LIGHT  7
#define MAT_GLASS  8
#define MAT_METAL  9
#define MAT_GLOBE 10

// ============================================================================
//  the frosted globe: a scattering medium with a filament inside
// ============================================================================
//  An opal light bulb: a dense, non-absorbing, isotropically scattering medium
//  filling a sphere, lit from within by a small bright core. Nothing about its
//  soft even glow is painted on -- the camera genuinely walks the medium, so the
//  globe comes out hot in the middle (short optical path to the core) and falls
//  off towards the rim, and it is the medium that turns a tiny fierce filament
//  into a large gentle source.
//
//  It stands on the floor in the far corner, behind the block, and it is the
//  room's MAIN light -- the ceiling panel is only fill. Two reasons: a pulse is
//  only visible if the pulsing source dominates the light budget, and a large
//  dim source keeps its radiance below the tonemap's shoulder, which is where
//  hue survives. It stays well inside the camera orbit: the eye must never end
//  up inside an emitter.
//
//  ENERGY. Interior rays see the walk; every non-delta vertex in the room gets
//  the globe by NEE against its surface (see globeLight) treating it as a
//  uniform emitter. Those two have to agree, and with no absorption they can be
//  made to agree exactly: all the core's power leaves through the globe's
//  surface, so
//      L_core * pi * 4pi r^2  =  L_globe * pi * 4pi R^2
//      L_core = L_globe * (R / r)^2
//  which is the ratio below. The filament is not made smaller than this on
//  purpose: that ratio goes as 1/r^2, and every bit of it lands on the variance
//  of any path that reaches the core through the medium (see CLAMP_INDIRECT). The one approximation left is angular: the walk's
//  exit radiance is centre-hot rather than uniform, so NEE has the total power
//  right and its distribution slightly wrong. That is invisible in the room's
//  lighting and only visible on the globe itself -- where you are looking at the
//  walk, not at NEE.
const vec3  GLOBE_C  = vec3( 1.55, 0.65, -1.90);   // y = GLOBE_R: on the floor
const float GLOBE_R  = 0.65;
const float CORE_R   = 0.16;                  // the filament
const float SIGMA_T  = 6.0;                   // extinction (1/m), all scattering
const float GLOBE_HUE = 15.0;                 // seconds per full hue rotation
const float BPM      = 120.0;
const float GLOBE_L_BASE = 1.0;               // steady surface radiance
const float GLOBE_L_KICK = 5.0;               // added at the peak of a hit

// Kick envelope, four-on-the-floor at 120 BPM: one hit per beat, the downbeat of
// each bar accented, plus a quieter ghost hit on the upbeat so it grooves rather
// than ticks. Shape is a kick drum's: attack of a couple of frames, decay over
// roughly a third of a beat. At 60 fps a beat is exactly 30 frames, so hits land
// on frame boundaries and a golden frame is reproducible.
float kickEnv(float time) {
    float beat = time * (BPM / 60.0);
    float hit = fract(beat);
    float accent = fract(beat / 4.0) < 0.25 ? 1.0 : 0.72;
    float e = smoothstep(0.0, 0.03, hit) * exp(-hit * 5.5) * accent;
    float up = fract(beat + 0.5);
    e += 0.28 * smoothstep(0.0, 0.03, up) * exp(-up * 9.0);
    return e;
}

// Hue rotation at constant luminance: only the chromaticity moves, so exposure
// does not have to chase the colour wheel around. The pulse rides on top.
vec3 globeRadiance(float time) {
    vec3 c = 0.5 + 0.5 * cos(TAU * time / GLOBE_HUE + vec3(0.0, 2.094, 4.189));
    c = pow(c, vec3(1.6));                    // deepen it, so the wash is a hue
    c /= max(lum(c), 0.12);
    return c * (GLOBE_L_BASE + GLOBE_L_KICK * kickEnv(time));
}
// The filament, from the energy balance above.
vec3 coreRadiance(float time) {
    return globeRadiance(time) * (GLOBE_R * GLOBE_R) / (CORE_R * CORE_R);
}

// The angular size of one pixel, used to pick a material filter footprint.
// Matching the lens keeps the checker filtered the same way at any resolution.
float pixelAngle(vec3 res) { return 1.0 / (FOCAL * res.y); }

// Floor and ceiling are both checkered, and in colour rather than grey.
// Two reasons beyond taste: a checker is the only thing in the room with
// high-frequency detail, which is what makes the glass ball read as glass at
// all (it magnifies whatever sits behind it), and tinted tiles put a second
// bounce colour into the room from above and below, not just from the walls.
// The ceiling pitch is coarser because it is only ever seen reflected,
// refracted, or at a grazing angle, and it is kept dark on purpose: it is the
// largest surface around the panel, so its albedo sets how much of the room is
// flat ambient fill. Dark up there buys contrast everywhere below, and makes the
// panel read as a source instead of as one more bright patch.
const float CHECK      = 0.45;                // floor tile pitch (m)
const float CHECK_CEIL = 0.90;                // ceiling tile pitch (m)
const vec3  FLOOR_A    = vec3(0.79, 0.75, 0.64);   // warm cream
const vec3  FLOOR_B    = vec3(0.07, 0.25, 0.28);   // deep teal
const vec3  CEIL_A     = vec3(0.40, 0.42, 0.47);   // cool slate
const vec3  CEIL_B     = vec3(0.11, 0.09, 0.18);   // near-black violet

// Box-filtered checker: instead of point sampling the square wave (which
// aliases into a shimmering mess towards the far wall, and would fight the
// denoiser for the rest of the frame) this integrates it analytically over the
// footprint fw, so distant tiles fade to their own average.
float checker(vec2 p, float fw, float pitch) {
    float w = max(fw, 1e-4) / pitch;
    vec2 q = p / pitch;
    vec2 i = 2.0 * (abs(fract((q - 0.5 * w) * 0.5) - 0.5)
                  - abs(fract((q + 0.5 * w) * 0.5) - 0.5)) / w;
    return 0.5 - 0.5 * i.x * i.y;
}

// Albedo and roughness. Floor and ceiling vary with position, but every caller
// passes both anyway: Buffer A demodulates by this value and the Image pass has
// to reproduce it exactly, so there must be one shared definition of it.
void materialAt(int mat, vec3 p, float fw, out vec3 alb, out float rough) {
    if (mat == MAT_FLOOR) {
        // slightly glazed, so the props get a soft reflection under them
        alb = mix(FLOOR_A, FLOOR_B, checker(p.xz, fw, CHECK));
        rough = 0.22;
    } else if (mat == MAT_CEIL) {
        alb = mix(CEIL_A, CEIL_B, checker(p.xz, fw, CHECK_CEIL));
        rough = 0.55;
    } else if (mat == MAT_RED) {
        alb = vec3(0.58, 0.09, 0.08); rough = 0.55;
    } else if (mat == MAT_GREEN) {
        alb = vec3(0.11, 0.46, 0.13); rough = 0.55;
    } else if (mat == MAT_BLUE) {
        alb = vec3(0.13, 0.24, 0.62); rough = 0.55;
    } else if (mat == MAT_AMBER) {
        alb = vec3(0.60, 0.42, 0.08); rough = 0.55;
    } else if (mat == MAT_BOX) {
        // deliberately neutral: it is the surface every other tint lands on
        alb = vec3(0.74, 0.73, 0.70); rough = 0.45;
    } else {
        // panel / glass / metal: never shaded through the diffuse path
        alb = vec3(1.0); rough = 0.50;
    }
}

// The albedo Buffer A divides out and the Image pass multiplies back in.
//
// Both derive it from the SAME primary-surface-replacement walk, and that is
// load-bearing: filtering irradiance only works if the two agree exactly. The
// obvious alternative -- each sample dividing by whatever albedo it happened to
// land on -- breaks down the moment a lens is in the way, because two samples a
// third of a pixel apart leave the glass ball pointed at different tiles. The
// divisor becomes noise that no amount of filtering can undo.
vec3 psrAlbedo(int mat, vec3 pos, float fw) {
    if (mat != MAT_FLOOR && mat != MAT_CEIL && mat != MAT_BOX &&
        mat != MAT_RED && mat != MAT_GREEN &&
        mat != MAT_BLUE && mat != MAT_AMBER) return vec3(1.0);
    vec3 alb; float rough;
    materialAt(mat, pos, fw, alb, rough);
    return alb;
}

struct Hit { float t; vec3 n; int mat; };

// The room, as seen from inside: the near slab is behind the ray, so the EXIT
// face of the slab test is the wall that gets hit -- no per-wall bounds test,
// and no way to miss. A ray that somehow started outside gets t <= 0 and is
// reported as a miss instead of hitting a back face.
void roomHit(vec3 ro, vec3 rd, inout Hit h) {
    vec3 inv = 1.0 / rd;                       // 1/0 = inf, which max() drops
    vec3 tlo = (vec3(-ROOM.x, 0.0, -ROOM.z) - ro) * inv;
    vec3 thi = (vec3( ROOM.x, ROOM.y, ROOM.z) - ro) * inv;
    vec3 tf = max(tlo, thi);
    float t = min(min(tf.x, tf.y), tf.z);
    if (t <= 1e-4 || t >= h.t) return;
    // inward normal of the exit face: the axis holding the minimum
    vec3 n = -sign(rd) * step(tf.xyz, tf.yzx) * step(tf.xyz, tf.zxy);
    vec3 p = ro + rd * t;
    int mat = MAT_CEIL;
    if (n.y > 0.5) {
        mat = MAT_FLOOR;
    } else if (n.y < -0.5) {
        // the panel fills a hole cut in the ceiling, so the two are coplanar
        // and there is no seam for a grazing ray to slip through
        if (abs(p.x) < LIGHT_R.x && abs(p.z) < LIGHT_R.y) mat = MAT_LIGHT;
    } else if (n.x > 0.5) {
        mat = MAT_RED;                         // -x wall
    } else if (n.x < -0.5) {
        mat = MAT_GREEN;                       // +x wall
    } else if (n.z > 0.5) {
        mat = MAT_BLUE;                        // -z wall
    } else {
        mat = MAT_AMBER;                       // +z wall
    }
    h.t = t; h.n = normalize(n); h.mat = mat;
}

// Sphere; the far root is taken when the origin is inside, which is how a ray
// travelling through the glass finds its exit. The normal always points out.
void sphereHit(vec3 ro, vec3 rd, vec3 c, float r, int mat, inout Hit h) {
    vec3 oc = ro - c;
    float b = dot(oc, rd);
    float cq = dot(oc, oc) - r * r;
    float disc = b * b - cq;
    if (disc <= 0.0) return;
    float sd = sqrt(disc);
    float t = -b - sd;
    if (t <= 1e-4) t = -b + sd;
    if (t <= 1e-4 || t >= h.t) return;
    h.t = t; h.n = (oc + rd * t) / r; h.mat = mat;
}

// Distance from a point INSIDE a sphere to its surface, and to the near surface
// of a sphere ahead (negative when missed). Both are used by the medium walk,
// which needs plain distances rather than a full Hit.
float sphereFar(vec3 ro, vec3 rd, vec3 c, float r) {
    vec3 oc = ro - c;
    float b = dot(oc, rd);
    return -b + sqrt(max(b * b - dot(oc, oc) + r * r, 0.0));
}
float sphereNear(vec3 ro, vec3 rd, vec3 c, float r) {
    vec3 oc = ro - c;
    float b = dot(oc, rd);
    float disc = b * b - dot(oc, oc) + r * r;
    if (disc <= 0.0) return -1.0;
    return -b - sqrt(disc);
}

mat2 rot2(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }

// Box with a yaw, tested in its own space; R is orthonormal, so the normal
// comes back with a transposed multiply (v * R == transpose(R) * v).
void boxHit(vec3 ro, vec3 rd, vec3 c, vec3 hx, float yaw, int mat, inout Hit h) {
    mat2 R = rot2(yaw);
    vec3 o = ro - c; o.xz = R * o.xz;
    vec3 d = rd;     d.xz = R * d.xz;
    vec3 inv = 1.0 / d;
    vec3 tlo = (-hx - o) * inv, thi = (hx - o) * inv;
    vec3 tn = min(tlo, thi), tf = max(tlo, thi);
    float tN = max(max(tn.x, tn.y), tn.z);
    float tF = min(min(tf.x, tf.y), tf.z);
    if (tN > tF || tN <= 1e-4 || tN >= h.t) return;
    vec3 n = -sign(d) * step(tn.yzx, tn.xyz) * step(tn.zxy, tn.xyz);
    n.xz = n.xz * R;
    h.t = tN; h.n = normalize(n); h.mat = mat;
}

Hit sceneHit(vec3 ro, vec3 rd, float tmax) {
    Hit h; h.t = tmax; h.n = vec3(0, 1, 0); h.mat = MAT_NONE;
    roomHit(ro, rd, h);
    sphereHit(ro, rd, GLASS_C, GLASS_R, MAT_GLASS, h);
    sphereHit(ro, rd, METAL_C, METAL_R, MAT_METAL, h);
    boxHit(ro, rd, BOX_C, BOX_H, BOX_ROT, MAT_BOX, h);
    boxHit(ro, rd, PLINTH_C, PLINTH_H, 0.0, MAT_BOX, h);
    sphereHit(ro, rd, GLOBE_C, GLOBE_R, MAT_GLOBE, h);
    if (h.t >= tmax) h.mat = MAT_NONE;
    return h;
}
// any-hit for shadow rays
bool sceneOccluded(vec3 ro, vec3 rd, float tmax) {
    Hit h = sceneHit(ro, rd, tmax);
    return h.mat != MAT_NONE;
}

// A uniformly distributed point on the ceiling panel, pulled a hair below the
// plane so a shadow ray aimed at it cannot graze the ceiling it sits in.
vec3 lightPoint(vec2 u) {
    return vec3((u.x * 2.0 - 1.0) * LIGHT_R.x,
                ROOM.y - 1e-3,
                (u.y * 2.0 - 1.0) * LIGHT_R.y);
}

// ============================================================================
//  BRDF
// ============================================================================
float frDielectric(float cosI, float eta) { // eta = n_transmitted / n_incident
    cosI = clamp(cosI, 0.0, 1.0);
    float s2 = (1.0 - cosI * cosI) / (eta * eta);
    if (s2 >= 1.0) return 1.0;               // TIR
    float cosT = sqrt(1.0 - s2);
    float rs = (cosI - eta * cosT) / (cosI + eta * cosT);
    float rp = (eta * cosI - cosT) / (eta * cosI + cosT);
    return 0.5 * (rs * rs + rp * rp);
}
float D_GGX(float NoH, float a2) {
    float d = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}
float G_smith(float NoV, float NoL, float a2) {
    float gv = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    float gl = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-7);
}
// full BRDF eval (diffuse + glazed specular F0 = 0.04)
vec3 evalBRDF(vec3 n, vec3 v, vec3 l, vec3 alb, float rough) {
    float NoL = dot(n, l), NoV = dot(n, v);
    if (NoL <= 0.0 || NoV <= 0.0) return vec3(0.0);
    vec3 hv = normalize(v + l);
    float NoH = clamp(dot(n, hv), 0.0, 1.0);
    float VoH = clamp(dot(v, hv), 0.0, 1.0);
    float a2 = rough * rough; a2 *= a2;
    float F = 0.04 + 0.96 * pow(1.0 - VoH, 5.0);
    float spec = D_GGX(NoH, a2) * G_smith(NoV, NoL, a2) * F;
    return alb / PI * (1.0 - F) + vec3(spec);
}
// GGX VNDF sampling (Heitz)
vec3 sampleGGX(vec3 n, vec3 v, float rough, vec2 u) {
    float a = rough * rough;
    mat3 B = orthoBasis(n);
    vec3 ve = v * B;                              // to tangent space
    vec3 vh = normalize(vec3(a * ve.x, a * ve.y, ve.z));
    float len2 = vh.x * vh.x + vh.y * vh.y;
    vec3 T1 = len2 > 0.0 ? vec3(-vh.y, vh.x, 0.0) / sqrt(len2) : vec3(1, 0, 0);
    vec3 T2 = cross(vh, T1);
    float r = sqrt(u.x), phi = TAU * u.y;
    float t1 = r * cos(phi), t2 = r * sin(phi);
    float s = 0.5 * (1.0 + vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;
    vec3 nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * vh;
    vec3 hv = normalize(vec3(a * nh.x, a * nh.y, max(0.0, nh.z)));
    return B * hv;                                // half vector, world space
}

// ============================================================================
//  primary hit with "primary surface replacement" (PSR)
// ============================================================================
//  The denoiser filters in screen space, so it needs a G-buffer that describes
//  what a pixel actually SHOWS. For a pixel covered by the glass or the mirror
//  ball that is not the ball: it is the wall or floor behind/inside it. So the
//  primary ray deterministically follows the specular chain and reports the
//  first non-delta surface, at the unfolded (virtual) distance along the ray.
//  Without this, refracted and reflected detail is smeared into mush.
void primaryHitFull(vec2 fragCoord, vec3 res, vec2 ang,
                    out float t, out vec3 n, out int mat,
                    out vec3 pos, out vec3 dir) {
    vec3 ro, rd;
    camRay(fragCoord, res, ang, ro, rd);
    bool inGlass = false;
    float tacc = 0.0;
    t = -1.0; n = vec3(0, 1, 0); mat = MAT_NONE; pos = ro; dir = rd;
    for (int i = 0; i < 3; i++) {
        Hit h = sceneHit(ro, rd, FAR);
        vec3 p = ro + rd * h.t;
        if (h.mat == MAT_GLASS) {
            vec3 nF = h.n;
            float eta = IOR_GLASS;
            if (inGlass) { nF = -h.n; eta = 1.0 / IOR_GLASS; }
            float F = frDielectric(clamp(dot(-rd, nF), 0.0, 1.0), eta);
            tacc += h.t;
            // Prefer the refracted branch: it carries the sharp, geometry-bound
            // detail, while the reflection is a smooth image that survives a
            // mismatched filter kernel. Above 0.75 there is nothing else left
            // (grazing angles, and total internal reflection).
            if (F > 0.75) {
                rd = reflect(rd, nF); ro = p + nF * 2e-4;
            } else {
                rd = normalize(refract(rd, nF, 1.0 / eta));
                ro = p - nF * 2e-4; inGlass = !inGlass;
            }
            // fallback, in case the chain never resolves within the loop
            t = tacc; n = h.n; mat = MAT_GLASS; pos = p; dir = rd;
            continue;
        }
        if (h.mat == MAT_METAL) {
            tacc += h.t;
            rd = reflect(rd, h.n); ro = p + h.n * 2e-4;
            t = tacc; n = h.n; mat = MAT_METAL; pos = p; dir = rd;
            continue;
        }
        t = tacc + h.t; n = h.n; mat = h.mat; pos = p; dir = rd;
        return;
    }
}
void primaryHit(vec2 fragCoord, vec3 res, vec2 ang,
                out float t, out vec3 n, out int mat) {
    vec3 pos, dir;
    primaryHitFull(fragCoord, res, ang, t, n, mat, pos, dir);
}

// ============================================================================
//  G-buffer packing: (depth, normal, material) -> one float (Buffer A alpha).
//  Traced once per pixel in Buffer A, reused by every denoiser tap, which is
//  far cheaper than re-tracing 25 taps per pass.
//  Layout (bit31..30 = 0 so the pattern is never NaN/Inf):
//  depth 14 bits (29..16) | oct.x 6 (15..10) | oct.y 6 (9..4) | mat+1 4 (3..0)
//  Depth quantum is 1/400 m over a 41 m range. The room is 8.4 m across, and an
//  unfolded specular chain through the glass ball can run to three segments of
//  that, so the range has to cover well past the room diagonal; 2.5 mm is still
//  far finer than the depth tolerance the wavelet passes compare against.
// ============================================================================
vec2 octWrap(vec2 v) {
    return (1.0 - abs(v.yx)) * vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}
float packGbuf(float t, vec3 n, int mat) {
    uint dz = t < 0.0 ? 16383u : uint(clamp(t * 400.0, 0.0, 16382.0));
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 e = n.y >= 0.0 ? n.xz : octWrap(n.xz);
    e = e * 0.5 + 0.5;
    uint ox = uint(e.x * 63.0 + 0.5), oy = uint(e.y * 63.0 + 0.5);
    uint bits = (dz << 16) | (ox << 10) | (oy << 4) | uint(mat + 1);
    return uintBitsToFloat(bits);
}
void unpackGbuf(float f, out float t, out vec3 n, out int mat) {
    uint bits = floatBitsToUint(f);
    uint dz = (bits >> 16) & 16383u;
    t = dz == 16383u ? -1.0 : float(dz) / 400.0;
    vec2 e = vec2(float((bits >> 10) & 63u), float((bits >> 4) & 63u)) / 63.0;
    e = e * 2.0 - 1.0;
    vec3 nn = vec3(e.x, 1.0 - abs(e.x) - abs(e.y), e.y);
    float tt = clamp(-nn.y, 0.0, 1.0);
    nn.x += nn.x >= 0.0 ? -tt : tt;
    nn.z += nn.z >= 0.0 ? -tt : tt;
    n = normalize(nn);
    mat = int(bits & 15u) - 1;
}

// ============================================================================
//  exposure & tonemap
// ============================================================================
//  The lighting never changes here, so exposure is a constant: one panel of
//  known radiance, no day cycle to adapt to.
const float EXPOSURE = 1.55;
vec3 ACES(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}
