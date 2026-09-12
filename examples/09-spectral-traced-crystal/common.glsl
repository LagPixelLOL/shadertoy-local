// Shared scene, optics, camera, and denoiser helpers.
// A full-height collimated slit illuminates a double-terminated pentagonal rod.
#define PI 3.14159265358979
#define TAU 6.28318530717959
const float FAR = 40.0;

const vec3 CAMERA_POS = vec3(0.0, 1.80, 3.95);
const float FOCAL = 0.78;
const float CRYSTAL_ROT = -0.215;

// Buffer A (0,0): orientation quaternion. (1,0): last mouse xy, button state
// (1 = released, 2 = pressed), and the frame at which accumulation restarted.
struct CrystalState { vec4 rotation; vec4 controls; };
CrystalState crystalState(sampler2D aCh) {
    return CrystalState(texelFetch(aCh, ivec2(0, 0), 0),
                        texelFetch(aCh, ivec2(1, 0), 0));
}
float accumulationCount(CrystalState state, int frame) {
    return max(float(frame) - state.controls.w + 1.0, 1.0);
}
CrystalState stepCrystal(CrystalState state, vec4 mouse, vec3 res, bool roll, int frame) {
    bool held = mouse.z > 0.0;
    float mag = dot(state.rotation, state.rotation);
    if (frame <= 0 || !(mag > 0.5 && mag < 1.5) || state.controls.z < 0.5) {
        state.rotation = vec4(0, sin(-0.5 * CRYSTAL_ROT), 0, cos(0.5 * CRYSTAL_ROT));
        state.controls = vec4(mouse.xy, held ? 2.0 : 1.0, float(frame));
        return state;
    }
    if (held && state.controls.z > 1.5) {
        vec2 drag = TAU * (mouse.xy - state.controls.xy) / res.xy;
        vec3 axis = roll ? vec3(0, 0, -drag.x) : vec3(-drag.y, drag.x, 0);
        float angle = length(axis);
        if (angle > 0.0) {
            vec4 dq = vec4(axis * (sin(0.5 * angle) / angle), cos(0.5 * angle));
            vec4 q = state.rotation;
            // Apply the incremental rotation in screen/world axes, not Euler
            // angles, so all orientations remain reachable without gimbal lock.
            state.rotation = normalize(vec4(dq.w * q.xyz + q.w * dq.xyz + cross(dq.xyz, q.xyz),
                                            dq.w * q.w - dot(dq.xyz, q.xyz)));
            state.controls.w = float(frame);
        }
    }
    state.controls.xyz = vec3(mouse.xy, held ? 2.0 : 1.0);
    return state;
}
void camRay(vec2 fc, vec3 res, out vec3 ro, out vec3 rd) {
    ro = CAMERA_POS;
    rd = normalize(vec3((fc - 0.5 * res.xy) / res.y, -FOCAL));
}

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
float rnd() { _rs = pcg4d(_rs); return float(_rs.x >> 8u) / 16777216.0; }
vec2 rnd2() { _rs = pcg4d(_rs); return vec2(_rs.xy >> 8u) / 16777216.0; }
float hash12(vec2 p) {
    return float(pcg4d(uvec4(floatBitsToUint(p), 0x2545F491u, 0x9E3779B9u)).x)
           / 4294967296.0;
}
// All accumulation/denoising buffers store CIE XYZ, not display RGB.
float lum(vec3 xyz) { return xyz.y; }

const float LAMBDA_MIN = 360.0;
const float LAMBDA_MAX = 830.0;
const float WAVELENGTH_PDF = 1.0 / (LAMBDA_MAX - LAMBDA_MIN); // per nm
const float D65_Y_INTEGRAL = 105.7255472031;

// CIE standard illuminant D65, sampled every 10 nm (360..830), divided by 100.
// Source: https://files.cie.co.at/CIE_std_illum_D65.csv (CIE, CC BY-SA 4.0).
const float D65[48] = float[48](
    0.466383, 0.520891, 0.499755, 0.546482, 0.827549, 0.914860,
    0.934318, 0.866823, 1.048650, 1.170080, 1.178120, 1.148610,
    1.159230, 1.088110, 1.093540, 1.078020, 1.047900, 1.076890,
    1.044050, 1.040460, 1.000000, 0.963342, 0.957880, 0.886856,
    0.900062, 0.895991, 0.876987, 0.832886, 0.836992, 0.800268,
    0.802146, 0.822778, 0.782842, 0.697213, 0.716091, 0.743490,
    0.616040, 0.698856, 0.750870, 0.635927, 0.464182, 0.668054,
    0.633828, 0.643040, 0.594519, 0.519590, 0.574406, 0.603125
);
float illuminantD65(float wavelength) {
    float t = clamp((wavelength - LAMBDA_MIN) / 10.0, 0.0, 47.0);
    int i = min(int(t), 46);
    return mix(D65[i], D65[i + 1], t - float(i));
}

// Analytic fit to the CIE 1931 2-degree color-matching functions:
// Wyman, Sloan & Shirley (2013), https://jcgt.org/published/0002/02/01/.
vec3 cieXYZ(float wavelength) {
    vec3 tx = (wavelength - vec3(442.0, 599.8, 501.1))
            * mix(vec3(0.0624, 0.0264, 0.0490), vec3(0.0374, 0.0323, 0.0382),
                  step(vec3(442.0, 599.8, 501.1), vec3(wavelength)));
    vec2 ty = (wavelength - vec2(568.8, 530.9))
            * mix(vec2(0.0213, 0.0613), vec2(0.0247, 0.0322),
                  step(vec2(568.8, 530.9), vec2(wavelength)));
    vec2 tz = (wavelength - vec2(437.0, 459.0))
            * mix(vec2(0.0845, 0.0385), vec2(0.0278, 0.0725),
                  step(vec2(437.0, 459.0), vec2(wavelength)));
    return vec3(dot(vec3(0.362, 1.056, -0.065), exp(-0.5 * tx * tx)),
                dot(vec2(0.821, 0.286), exp(-0.5 * ty * ty)),
                dot(vec2(1.217, 0.681), exp(-0.5 * tz * tz)));
}
float glassIOR(float wavelength) {
    // Cauchy dispersion model; coefficients use wavelength in micrometres.
    float um = wavelength * 0.001;
    return 1.475 + 0.014 / (um * um);
}
float glassAbsorption(float wavelength) {
    // A weak red absorption band, in inverse metres, leaves a nearly clear rod.
    float band = (wavelength - 690.0) / 65.0;
    return 0.012 + 0.050 * exp(-0.5 * band * band);
}
vec3 xyzToLinearSRGB(vec3 xyz) {
    return vec3(dot(vec3( 3.24096994, -1.53738318, -0.49861076), xyz),
                dot(vec3(-0.96924364,  1.87596750,  0.04155506), xyz),
                dot(vec3( 0.05563008, -0.20397696,  1.05697151), xyz));
}
mat3 orthoBasis(vec3 n) {
    vec3 t = abs(n.y) < 0.99 ? normalize(cross(n, vec3(0, 1, 0))) : vec3(1, 0, 0);
    return mat3(t, cross(n, t), n);
}
vec3 cosineDir(vec3 n, vec2 u) {
    float r = sqrt(u.x), ph = TAU * u.y;
    return orthoBasis(n) * vec3(r * cos(ph), r * sin(ph), sqrt(1.0 - u.x));
}
vec3 sampleCone(vec3 d, float cosAng, vec2 u) {
    float ca = mix(cosAng, 1.0, u.x);
    float sa = sqrt(max(1.0 - ca * ca, 0.0));
    return orthoBasis(d) * vec3(cos(TAU * u.y) * sa, sin(TAU * u.y) * sa, ca);
}

const vec3 ROOM = vec3(4.20, 3.60, 4.20);
const float LIGHT_HALF_W = 0.085;
const float LIGHT_A = 2.0 * LIGHT_HALF_W * ROOM.y;
// Both components of the same bar share the D65 spectrum, applied in tracePath.
const float LIGHT_E = 3.2;                       // weak off-axis diffuser
const float BEAM_E = 65.0;                       // integrated cone radiance
const float BEAM_ANGLE = 0.03;                   // half-angle in radians
const float HAZE = 0.03;                         // extinction per metre

const float CRYSTAL_R = 0.57;
const float CRYSTAL_H = 0.88;                    // straight section half-height
const float CRYSTAL_CAP = 0.55;
const vec3 CRYSTAL_C = vec3(0.0, CRYSTAL_H + CRYSTAL_CAP, 0.0);
mat3 crystalRotation = mat3(1.0);
void setCrystalRotation(vec4 q) {
    vec3 s = 2.0 * q.xyz;
    float xx = q.x * s.x, yy = q.y * s.y, zz = q.z * s.z;
    float xy = q.x * s.y, xz = q.x * s.z, yz = q.y * s.z;
    float wx = q.w * s.x, wy = q.w * s.y, wz = q.w * s.z;
    crystalRotation = mat3(1.0 - yy - zz, xy + wz, xz - wy,
                           xy - wz, 1.0 - xx - zz, yz + wx,
                           xz + wy, yz - wx, 1.0 - xx - yy);
}

#define MAT_NONE -1
#define MAT_FLOOR 0
#define MAT_CEIL 1
#define MAT_XMIN 2
#define MAT_XMAX 3
#define MAT_ZMIN 4
#define MAT_ZMAX 5
#define MAT_LIGHT 6
#define MAT_GLASS 7

float pixelAngle(vec3 res) { return 1.0 / (FOCAL * res.y); }
void materialAt(int mat, vec3 p, vec2 fw, out float alb, out float rough) {
    alb = 0.78; // spectrally flat reflectance for every white wall
    rough = 0.45;
}
float psrAlbedo(int mat, vec3 p, vec2 fw) {
    if (mat < MAT_FLOOR || mat > MAT_ZMAX) return 1.0;
    float alb, rough;
    materialAt(mat, p, fw, alb, rough);
    return alb;
}

struct Hit { float t; vec3 n; int mat; };

// The solid is the intersection of five side planes and two five-face pyramids.
vec4 crystalPlane(int i) {
    float a = TAU * float(i % 5) / 5.0;
    float apothem = CRYSTAL_R * cos(PI / 5.0);
    if (i < 5) return vec4(crystalRotation * vec3(cos(a), 0.0, sin(a)), apothem);
    float k = apothem / CRYSTAL_CAP;
    vec3 n = vec3(cos(a), i < 10 ? k : -k, sin(a));
    return vec4(crystalRotation * n, k * (CRYSTAL_H + CRYSTAL_CAP)) / length(n);
}
bool clipPlane(vec3 ro, vec3 rd, vec4 plane, inout float lo, inout float hi) {
    float d = dot(rd, plane.xyz);
    float s = plane.w - dot(ro, plane.xyz);
    if (abs(d) < 1e-7) return s >= -1e-5;
    float t = s / d;
    if (d < 0.0) lo = max(lo, t); else hi = min(hi, t);
    return lo <= hi;
}
bool clipCrystal(vec3 ro, vec3 rd, int onFace, inout float lo, inout float hi) {
    for (int i = 0; i < 15; i++) {
        // Projection already puts this whole line on the face. Testing it again
        // turns roundoff in a zero denominator into an arbitrary interval cut.
        if (i == onFace) continue;
        if (!clipPlane(ro - CRYSTAL_C, rd, crystalPlane(i), lo, hi)) return false;
    }
    return true;
}
void crystalHit(vec3 ro, vec3 rd, inout Hit h) {
    // Test bounds in local space; a tilted rod extends beyond its upright AABB.
    vec3 o = ro - CRYSTAL_C;
    vec3 ext = vec3(CRYSTAL_R, CRYSTAL_H + CRYSTAL_CAP, CRYSTAL_R);
    vec3 localO = o * crystalRotation;
    vec3 inv = 1.0 / (rd * crystalRotation);
    vec3 a = (-ext - localO) * inv, b = (ext - localO) * inv;
    vec3 mn = min(a, b), mx = max(a, b);
    if (max(max(mn.x, mn.y), max(mn.z, 0.0)) >
        min(min(mx.x, mx.y), min(mx.z, h.t))) return;
    float lo = -FAR, hi = FAR;
    vec3 nlo = vec3(0), nhi = vec3(0);
    for (int i = 0; i < 15; i++) {
        vec4 pl = crystalPlane(i);
        float d = dot(rd, pl.xyz), s = pl.w - dot(o, pl.xyz);
        if (abs(d) < 1e-7) { if (s < 0.0) return; continue; }
        float t = s / d;
        if (d < 0.0 && t > lo) { lo = t; nlo = pl.xyz; }
        if (d > 0.0 && t < hi) { hi = t; nhi = pl.xyz; }
        if (lo > hi) return;
    }
    float t = lo > 1e-4 ? lo : hi;
    if (t <= 1e-4 || t >= h.t) return;
    h.t = t; h.n = lo > 1e-4 ? nlo : nhi; h.mat = MAT_GLASS;
}
void roomHit(vec3 ro, vec3 rd, inout Hit h) {
    vec3 inv = 1.0 / rd;
    vec3 a = (vec3(-ROOM.x, 0.0, -ROOM.z) - ro) * inv;
    vec3 b = (ROOM - ro) * inv;
    vec3 tf = max(a, b);
    float t = min(min(tf.x, tf.y), tf.z);
    if (t <= 1e-4 || t >= h.t) return;
    vec3 n = -sign(rd) * step(tf.xyz, tf.yzx) * step(tf.xyz, tf.zxy);
    vec3 p = ro + rd * t;
    int mat = MAT_CEIL;
    if (n.y > 0.5) mat = MAT_FLOOR;
    else if (n.y < -0.5) mat = MAT_CEIL;
    else if (n.x > 0.5) mat = abs(p.z) < LIGHT_HALF_W ? MAT_LIGHT : MAT_XMIN;
    else if (n.x < -0.5) mat = MAT_XMAX;
    else if (n.z > 0.5) mat = MAT_ZMIN;
    else mat = MAT_ZMAX;
    h.t = t; h.n = normalize(n); h.mat = mat;
}
Hit sceneHit(vec3 ro, vec3 rd, float tmax) {
    Hit h; h.t = tmax; h.n = vec3(0, 1, 0); h.mat = MAT_NONE;
    roomHit(ro, rd, h);
    crystalHit(ro, rd, h);
    return h;
}
bool sceneOccluded(vec3 ro, vec3 rd, float tmax) {
    return sceneHit(ro, rd, tmax).mat != MAT_NONE;
}
vec3 lightPoint(vec2 u) {
    return vec3(-ROOM.x + 1e-3, u.x * ROOM.y, (2.0 * u.y - 1.0) * LIGHT_HALF_W);
}

float frDielectric(float cosI, float eta) {
    cosI = clamp(cosI, 0.0, 1.0);
    float s2 = (1.0 - cosI * cosI) / (eta * eta);
    if (s2 >= 1.0) return 1.0;
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
float evalBRDF(vec3 n, vec3 v, vec3 l, float alb, float rough) {
    float NoL = dot(n, l), NoV = dot(n, v);
    if (NoL <= 0.0 || NoV <= 0.0) return 0.0;
    vec3 hv = normalize(v + l);
    float F = 0.04 + 0.96 * pow(1.0 - clamp(dot(v, hv), 0.0, 1.0), 5.0);
    float a2 = rough * rough; a2 *= a2;
    float spec = D_GGX(clamp(dot(n, hv), 0.0, 1.0), a2)
                 * G_smith(NoV, NoL, a2) * F;
    return alb / PI * (1.0 - F) + spec;
}
vec3 sampleGGX(vec3 n, vec3 v, float rough, vec2 u) {
    float a = rough * rough;
    mat3 B = orthoBasis(n);
    vec3 ve = v * B;
    vec3 vh = normalize(vec3(a * ve.x, a * ve.y, ve.z));
    float len2 = vh.x * vh.x + vh.y * vh.y;
    vec3 T1 = len2 > 0.0 ? vec3(-vh.y, vh.x, 0.0) / sqrt(len2) : vec3(1, 0, 0);
    vec3 T2 = cross(vh, T1);
    float r = sqrt(u.x), phi = TAU * u.y;
    float t1 = r * cos(phi), t2 = r * sin(phi);
    float s = 0.5 * (1.0 + vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;
    vec3 nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * vh;
    return B * normalize(vec3(a * nh.x, a * nh.y, max(0.0, nh.z)));
}

// Follow the transmitted image so the original temporal/wavelet denoiser can
// retain the wall lighting seen through the planar optical interfaces. Only this
// deterministic denoiser guide uses 550 nm; radiance paths sample the spectrum.
void primaryHitFull(vec2 fc, vec3 res, out float t, out vec3 n,
                    out int mat, out vec3 pos, out vec3 dir) {
    vec3 ro, rd; camRay(fc, res, ro, rd);
    bool inGlass = false;
    float tacc = 0.0;
    t = -1.0; n = vec3(0, 1, 0); mat = MAT_NONE; pos = ro; dir = rd;
    for (int i = 0; i < 8; i++) {
        Hit h = sceneHit(ro, rd, FAR);
        if (h.mat == MAT_NONE) return;
        vec3 p = ro + rd * h.t;
        tacc += h.t;
        t = tacc; n = h.n; mat = h.mat; pos = p; dir = rd;
        if (h.mat != MAT_GLASS) return;
        vec3 nf = inGlass ? -h.n : h.n;
        float ior = glassIOR(550.0);
        float eta = inGlass ? 1.0 / ior : ior;
        float f = frDielectric(dot(-rd, nf), eta);
        if (f > 0.75) { rd = reflect(rd, nf); ro = p + nf * 2e-4; }
        else {
            rd = normalize(refract(rd, nf, 1.0 / eta));
            ro = p - nf * 2e-4; inGlass = !inGlass;
        }
    }
}
vec2 psrFootprint(vec2 fc, vec3 res, float tC, vec3 nC,
                  int matC, vec3 posC, vec3 dirC) {
    float pinhole = max(tC, 0.0) * pixelAngle(res) / max(abs(dot(nC, dirC)), 0.45);
    float t1; vec3 n1; int m1; vec3 pxp, d1;
    primaryHitFull(fc + vec2(0.5, 0), res, t1, n1, m1, pxp, d1);
    float t2; vec3 n2; int m2; vec3 pxm, d2;
    primaryHitFull(fc - vec2(0.5, 0), res, t2, n2, m2, pxm, d2);
    float t3; vec3 n3; int m3; vec3 pyp, d3;
    primaryHitFull(fc + vec2(0, 0.5), res, t3, n3, m3, pyp, d3);
    float t4; vec3 n4; int m4; vec3 pym, d4;
    primaryHitFull(fc - vec2(0, 0.5), res, t4, n4, m4, pym, d4);
    vec3 du = vec3(0), dv = vec3(0);
    if (m1 == matC && m2 == matC) du = pxp - pxm;
    if (m3 == matC && m4 == matC) dv = pyp - pym;
    vec2 fw = abs(du.xz) + abs(dv.xz);
    return max(fw, vec2(dot(fw, fw) > 1e-12 ? 0.0 : pinhole));
}

vec2 octWrap(vec2 v) {
    return (1.0 - abs(v.yx)) * vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}
float packGbuf(float t, vec3 n, int mat) {
    uint dz = t < 0.0 ? 16383u : uint(clamp(t * 400.0, 0.0, 16382.0));
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    vec2 e = (n.y >= 0.0 ? n.xz : octWrap(n.xz)) * 0.5 + 0.5;
    uint ox = uint(e.x * 63.0 + 0.5), oy = uint(e.y * 63.0 + 0.5);
    return uintBitsToFloat((dz << 16) | (ox << 10) | (oy << 4) | uint(mat + 1));
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
    n = normalize(nn); mat = int(bits & 15u) - 1;
}
const float EXPOSURE = 2.1;
vec3 ACES(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
