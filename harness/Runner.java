import com.elfmcys.yesstevemodel.geckolib3.geo.NativeModelRenderer;
import com.elfmcys.yesstevemodel.geckolib3.geo.render.built.GeoModel;
import mock.MockBufferBuilder;
import mock.MockVertexFormat;

import java.io.PrintWriter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * Differential harness runner.
 *
 * Usage:   java Runner <case> <outFile>
 *          YSM_LIB   = absolute path of the libysm-core.so under test.
 *          YSM_SEMANTIC=1 -> semantic mode: real assertions against the
 *                            patched behaviour (offset9 / non-tree fixes);
 *                            exits 1 on mismatch. Golden diverges by design.
 *
 * TWO wire protocols, matching the two production Java writers:
 *   SIMD wire  (GeoModel.buildNativeCache -> nInitModelCache):
 *       4 + bones*25 + cubes*5 + quads*93 bytes; per quad: translucent byte,
 *       12 float positions, 8 float uvs, 3 float normal.
 *   GPU wire   (GpuMeshBuilder.serializeModel -> nBuildGpuMesh):
 *       same minus the translucent byte = 92 bytes per quad.
 *       Evidence: shipped .so GM_nBuildGpuMesh.asm quad advance = 0x5c (92B);
 *       fed the 93B wire the shipped binary desyncs and crashes.
 *
 * All native-facing ByteBuffers MUST be allocateDirect: GetDirectBufferAddress
 * returns null for heap buffers, which silently voids a case (handle=0
 * early-return on both sides = vacuous green). Found 2026-09-17 when this
 * harness was re-validated: its original 10/10 "green" had never executed a
 * single native instruction (heap ByteBuffers).
 */
public class Runner {

    // ----- tiny model DSL -------------------------------------------------

    record Quad(boolean translucent, float[] positions, float[] uvs, float[] normal) {}

    record Cube(boolean cullable, List<Quad> quads) {}

    record Bone(int parent, int partMask, boolean glow, float px, float py, float pz,
                List<Cube> cubes) {}

    static Quad quad(boolean translucent, boolean cullOrderUnused, float z, float u0) {
        // CCW square (det > 0 under the test projection).
        float[] pos = {
                0, 0, z, 1, 0, z, 1, 1, z, 0, 1, z
        };
        float[] uv = {u0, 0, u0 + 0.25f, 0, u0 + 0.25f, 0.25f, u0, 0.25f};
        float[] n = {0, 0, 1};
        return new Quad(translucent, pos, uv, n);
    }

    static float[] f(float... v) {
        return v;
    }

    // ----- case definitions -------------------------------------------------

    private static List<Bone> bonesFor(String testCase) {
        List<Bone> bones = new ArrayList<>();
        switch (testCase) {
            case "normal" -> {
                // root + child, cullable opaque quads, no translucency,
                // no skip flags, no state sentinel.
                bones.add(new Bone(-1, 0, false, 0, 0, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5f, 0f))))));
                bones.add(new Bone(0, 0, false, 4, 2, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5.5f, 0.5f))))));
            }
            case "translucent" -> {
                // One bone carrying an opaque cullable, a translucent cullable
                // and a translucent non-cullable quad -> exercises all four
                // buckets.
                bones.add(new Bone(-1, 0, false, 0, 0, 0, List.of(
                        new Cube(true, List.of(quad(false, true, 5f, 0f))),
                        new Cube(true, List.of(quad(true, true, 5f, 0.25f))))));
                bones.add(new Bone(0, 0, false, 4, 2, 0, List.of(
                        new Cube(false, List.of(quad(true, false, 6f, 0.5f))))));
            }
            case "glow" -> {
                bones.add(new Bone(-1, 0, true, 0, 0, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5f, 0f))))));
                bones.add(new Bone(0, 0, false, 4, 2, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5.5f, 0.5f))))));
            }
            case "partMask" -> {
                bones.add(new Bone(-1, 1, false, 0, 0, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5f, 0f))))));
                bones.add(new Bone(0, 2, false, 4, 2, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5.5f, 0.5f))))));
                bones.add(new Bone(0, 3, false, 8, 2, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 6f, 0.75f))))));
            }
            case "hiddenSubtree" -> {
                // Bone 1 flagged skip-children -> bone 2's quads disappear.
                bones.add(new Bone(-1, 0, false, 0, 0, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5f, 0f))))));
                bones.add(new Bone(0, 0, false, 4, 2, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5.5f, 0.5f))))));
                bones.add(new Bone(1, 0, false, 8, 2, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 6f, 0.75f))))));
            }
            case "emptyMesh" -> {
                bones.add(new Bone(-1, 0, false, 0, 0, 0, List.of()));
                bones.add(new Bone(0, 0, false, 4, 2, 0, List.of(
                        new Cube(true, List.of()))));
            }
            case "state" -> {
                // Sentinel animData[+11] = 1.0 on both bones -> native fills
                // the state array.
                bones.add(new Bone(-1, 0, false, 1, 2, 3,
                        List.of(new Cube(true, List.of(quad(false, true, 5f, 0f))))));
                bones.add(new Bone(0, 0, false, 4, 2, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5.5f, 0.5f))))));
            }
            case "zeroScale" -> {
                // Probe: bone 1 has zero scale. Distinguishes "skip subtree on
                // scale==0" (upstream behaviour) from "no scale check" (what
                // the shipped binary disassembly shows).
                bones.add(new Bone(-1, 0, false, 0, 0, 0,
                        List.of(new Cube(false, List.of(quad(false, false, 5f, 0f))))));
                bones.add(new Bone(0, 0, false, 4, 2, 0,
                        List.of(new Cube(false, List.of(quad(false, false, 5.5f, 0.5f))))));
            }
            case "culled" -> {
                // Projection enables frustum culling; bone 1 sits behind the
                // camera plane and must be culled.
                bones.add(new Bone(-1, 0, false, 0, 0, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5f, 0f))))));
                bones.add(new Bone(0, 0, false, 0, 0, 0,
                        List.of(new Cube(true, List.of(new Quad(false,
                                f(0, 0, -50f, 1, 0, -50f, 1, 1, -50f, 0, 1, -50f),
                                f(0f, 0f, .25f, 0f, .25f, .25f, 0f, .25f),
                                f(0, 0, 1)))))));
            }
            case "gpuMultiQuad" -> {
                // GPU-protocol drift case: multi-bone/multi-cube/multi-quad,
                // pins the 92B quad stride byte-level (a 93B reader desyncs
                // after the first quad).
                bones.add(new Bone(-1, 1, true, 1, 2, 3, List.of(
                        new Cube(true, List.of(quad(false, true, 5f, 0f))),
                        new Cube(true, List.of(quad(true, true, 5f, 0.25f))),
                        new Cube(false, List.of(quad(false, false, 5.25f, 0.5f))))));
                bones.add(new Bone(0, 2, false, 4, 2, 0, List.of(
                        new Cube(false, List.of(quad(true, false, 6f, 0.5f))),
                        new Cube(true, List.of(quad(false, true, 6.5f, 0.75f))))));
                bones.add(new Bone(1, 2, false, 2, 1, 0, List.of(
                        new Cube(true, List.of(quad(true, true, 7f, 0.1f))))));
            }
            // ---- semantic cases (offset9 fix; diverge from shipped by design) ----
            case "offset9Self" -> {
                // b1 hidden itself -> own quads culled, parent chain intact.
                bones.add(new Bone(-1, 0, false, 0, 0, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5f, 0f))))));
                bones.add(new Bone(0, 0, false, 4, 2, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5.5f, 0.5f))))));
            }
            case "offset9Root" -> {
                // root hidden -> whole subtree culled (CPU visibleCache parity).
                bones.add(new Bone(-1, 0, false, 0, 0, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5f, 0f))))));
                bones.add(new Bone(0, 0, false, 4, 2, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5.5f, 0.5f))))));
                bones.add(new Bone(1, 0, false, 8, 2, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 6f, 0.75f))))));
            }
            case "offset9Midtree" -> {
                // b1 hidden: b0 + b3 (second child of root) visible,
                // b1 + its child b2 culled.
                bones.add(new Bone(-1, 0, false, 0, 0, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5f, 0f))))));
                bones.add(new Bone(0, 0, false, 4, 2, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5.5f, 0.5f))))));
                bones.add(new Bone(1, 0, false, 8, 2, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 6f, 0.75f))))));
                bones.add(new Bone(0, 0, false, 2, 0, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 6.5f, 0.9f))))));
            }
            case "offset9PlusSkip10" -> {
                // offset9 and offset10 both set on the root: single k-advance,
                // no double-skip, whole subtree gone.
                bones.add(new Bone(-1, 0, false, 0, 0, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5f, 0f))))));
                bones.add(new Bone(0, 0, false, 4, 2, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5.5f, 0.5f))))));
            }
            case "offset9GpuOnly" -> {
                // GPU leg: offset9 on b1; self + inherited isHidden asserted.
                bones.add(new Bone(-1, 0, false, 0, 0, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5f, 0f))))));
                bones.add(new Bone(0, 0, false, 4, 2, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5.5f, 0.5f))))));
                bones.add(new Bone(1, 0, false, 8, 2, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 6f, 0.75f))))));
            }
            // ---- semantic cases (non-tree defense; shipped crashes) ----
            case "nontreeBadParent" -> {
                // parentIdx 42 out of range: shipped OOB-writes children[42],
                // patched clamps to root.
                bones.add(new Bone(-1, 0, false, 0, 0, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5f, 0f))))));
                bones.add(new Bone(42, 0, false, 4, 2, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5.5f, 0.5f))))));
            }
            case "nontreeCycle" -> {
                // 2-cycle: no root -> shipped evalOrder empty -> render loop
                // reads evalOrder[k] OOB; patched appends leftovers.
                bones.add(new Bone(1, 0, false, 0, 0, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5f, 0f))))));
                bones.add(new Bone(0, 0, false, 4, 2, 0,
                        List.of(new Cube(true, List.of(quad(false, true, 5.5f, 0.5f))))));
            }
            default -> throw new IllegalArgumentException("unknown case " + testCase);
        }
        return bones;
    }

    private static final java.util.Set<String> OFFSET9_CASES = java.util.Set.of(
            "offset9Self", "offset9Root", "offset9Midtree", "offset9PlusSkip10", "offset9GpuOnly");

    private static float[] animFor(String testCase, int boneCount) {
        float[] anim = new float[boneCount * 12];
        boolean isOffset9 = OFFSET9_CASES.contains(testCase);
        for (int b = 0; b < boneCount; ++b) {
            int o = b * 12;
            anim[o + 0] = 0.3f + 0.1f * b;   // rx
            anim[o + 1] = 0.5f;              // ry
            anim[o + 2] = 0.2f;              // rz
            anim[o + 3] = 2f + b;            // tx
            anim[o + 4] = 1f;                // ty
            anim[o + 5] = 0.5f;              // tz
            anim[o + 6] = 1.1f;              // sx
            anim[o + 7] = 0.9f;              // sy
            anim[o + 8] = 1.0f;              // sz
            anim[o + 9] = 0f;                // offset9 HIDDEN
            anim[o + 10] = "hiddenSubtree".equals(testCase) && b == 1 ? 1f : 0f; // offset10 skip flag
            anim[o + 11] = "state".equals(testCase) ? 1f : 0f;                   // state sentinel
        }
        if ("zeroScale".equals(testCase)) {
            anim[12 + 6] = 0f;
            anim[12 + 7] = 0f;
            anim[12 + 8] = 0f;
        }
        if (isOffset9) {
            switch (testCase) {
                case "offset9Self" -> anim[12 + 9] = 1f;
                case "offset9Root" -> anim[9] = 1f;
                case "offset9Midtree" -> anim[12 + 9] = 1f;
                case "offset9PlusSkip10" -> {
                    anim[9] = 1f;
                    anim[10] = 1f; // offset10 on the same root bone
                }
                case "offset9GpuOnly" -> anim[12 + 9] = 1f;
            }
        }
        return anim;
    }

    private static float[] matrices(boolean culling) {
        // 48 floats: pose(16) + normal(16) + proj(16), column-major like
        // Matrix4f#get(float[], int).
        float[] m = new float[48];
        m[0] = m[5] = m[10] = m[15] = 1f;             // pose = identity
        m[16 + 0] = m[16 + 5] = m[16 + 10] = m[16 + 15] = 1f; // normal = identity
        if (culling) {
            // proj: w = z (perspective-ish), enables culling (|m[11]| > 1e-3).
            m[32 + 0] = 1f;
            m[32 + 5] = 1f;
            m[32 + 10] = 1f;
            m[32 + 11] = 1f;
        } else {
            m[32 + 0] = 1f;
            m[32 + 5] = 1f;
            m[32 + 10] = 1f;
            m[32 + 15] = 1f;                          // proj = identity
        }
        return m;
    }

    // ----- wire buffers -------------------------------------------------------

    /** SIMD wire: mirrors GeoModel.buildNativeCache() — translucent byte per quad (93B). */
    private static ByteBuffer simdWire(List<Bone> bones) {
        return wire(bones, true);
    }

    /** GPU wire: mirrors GpuMeshBuilder.serializeModel() — no translucent byte (92B). */
    private static ByteBuffer gpuWire(List<Bone> bones) {
        return wire(bones, false);
    }

    private static ByteBuffer wire(List<Bone> bones, boolean translucentByte) {
        int quads = 0;
        for (Bone b : bones) for (Cube c : b.cubes()) quads += c.quads().size();
        int cubes = 0;
        for (Bone b : bones) cubes += b.cubes().size();
        int quadSize = translucentByte ? 93 : 92;
        int size = 4 + bones.size() * 25 + cubes * 5 + quads * quadSize;
        // MUST be direct: GetDirectBufferAddress returns null for heap buffers.
        ByteBuffer buf = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder());
        buf.putInt(bones.size());
        for (Bone b : bones) {
            buf.putInt(b.parent());
            buf.putInt(b.partMask());
            buf.put((byte) (b.glow() ? 1 : 0));
            buf.putFloat(b.px());
            buf.putFloat(b.py());
            buf.putFloat(b.pz());
            buf.putInt(b.cubes().size());
            for (Cube c : b.cubes()) {
                buf.put((byte) (c.cullable() ? 1 : 0));
                buf.putInt(c.quads().size());
                for (Quad q : c.quads()) {
                    if (translucentByte) buf.put((byte) (q.translucent() ? 1 : 0));
                    for (float p : q.positions()) buf.putFloat(p);
                    for (float u : q.uvs()) buf.putFloat(u);
                    for (float n : q.normal()) buf.putFloat(n);
                }
            }
        }
        buf.position(0);
        return buf;
    }

    // ----- dump helpers -----------------------------------------------------

    private static String hex(ByteBuffer buf, int len) {
        StringBuilder sb = new StringBuilder();
        ByteBuffer d = buf.duplicate().order(ByteOrder.nativeOrder());
        d.position(0);
        d.limit(Math.min(len, d.capacity()));
        for (int i = 0; i < d.limit(); i++) {
            sb.append(String.format("%02x", d.get(i)));
        }
        return sb.toString();
    }

    private static String hex(int[] arr, int len) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < len; i++) sb.append(String.format("%08x", arr[i]));
        return sb.toString();
    }

    private static String hexFloat(float[] arr) {
        StringBuilder sb = new StringBuilder();
        for (float v : arr) sb.append(String.format("%08x", Float.floatToRawIntBits(v)));
        return sb.toString();
    }

    /** isHidden flag of BoneDataOut (144B stride: transform 64 + normal 64 + light 4 + isHidden 4). */
    private static int isHiddenAt(ByteBuffer outBone, int boneIdx) {
        return outBone.getInt(boneIdx * 144 + 132);
    }

    // ----- driver -------------------------------------------------------------

    // captured for semantic assertions
    static int slowCount, fastVertices, maskCount;
    static int[] gpuHidden;
    static boolean cacheAlive, gpuAlive;

    public static void main(String[] args) throws Exception {
        String testCase = args[0];
        String outPath = args[1];
        boolean semantic = "1".equals(System.getenv("YSM_SEMANTIC"));
        PrintWriter out = new PrintWriter(outPath, "UTF-8");

        List<Bone> bones = bonesFor(testCase);
        int boneCount = bones.size();
        float[] anim = animFor(testCase, boneCount);

        // Register fast-path handles against the mocks.
        GeoModel.nInitSIMD(MockBufferBuilder.class, "buffer", "vertices", "nextElementByte",
                "ensureCapacity", "mode", MockVertexFormat.Mode.class);

        long handle = GeoModel.nInitModelCache(simdWire(bones));
        cacheAlive = handle != 0;
        out.println("handle!=" + cacheAlive);
        out.flush();
        if (handle == 0) {
            if (semantic) out.println("semantic.result=FAIL (cache handle=0)");
            out.close();
            if (semantic) System.exit(1);
            return;
        }

        float[] mats = matrices("culled".equals(testCase));
        int light = 0x12345678;
        int overlay = 0x00A0B0C0;
        float[] state = new float[boneCount * 4];

        // --- slow path (submitVertices callback) ---
        NativeModelRenderer.callCount = 0;
        GeoModel.nComputeModelVertices(handle, null, mats, anim, state,
                0, light, overlay, 0.25f, 0.5f, 0.75f, 1.0f);
        slowCount = NativeModelRenderer.lastVertexCount;
        out.println("slow.calls=" + NativeModelRenderer.callCount);
        out.println("slow.count=" + slowCount);
        if (NativeModelRenderer.lastFBuf != null && slowCount > 0) {
            int fLen = slowCount * 12 * 4;
            int iLen = slowCount * 2 * 4;
            out.println("slow.f=" + hex(NativeModelRenderer.lastFBuf, fLen));
            out.println("slow.i=" + hex(NativeModelRenderer.lastIBuf, iLen));
        }

        // --- state array after slow pass ---
        out.println("state=" + hexFloat(state));

        // --- fast path (mock BufferBuilder direct write) ---
        MockBufferBuilder builder = new MockBufferBuilder(4096);
        java.util.Arrays.fill(state, 0f);
        GeoModel.nComputeModelVertices(handle, builder, mats, anim, state,
                0, light, overlay, 0.25f, 0.5f, 0.75f, 1.0f);
        fastVertices = builder.vertices;
        out.println("fast.ensureCalls=" + builder.ensureCapacityCalls);
        out.println("fast.ensureArg=" + builder.lastEnsureCapacityArg);
        out.println("fast.written=" + builder.nextElementByte);
        out.println("fast.buffer=" + hex(builder.buffer, Math.max(0, builder.nextElementByte)));
        out.println("fast.vertices=" + fastVertices);

        // --- partMask pass (slow only; mask=2) ---
        NativeModelRenderer.callCount = 0;
        GeoModel.nComputeModelVertices(handle, null, mats, anim, state,
                2, light, overlay, 1f, 1f, 1f, 1f);
        maskCount = NativeModelRenderer.lastVertexCount;
        out.println("mask.calls=" + NativeModelRenderer.callCount);
        out.println("mask.count=" + maskCount);
        if (maskCount > 0) {
            out.println("mask.f=" + hex(NativeModelRenderer.lastFBuf,
                    maskCount * 12 * 4));
        }
        out.flush();

        // --- GPU mesh path (92B GPU wire, mirrors GpuMeshBuilder.serializeModel) ---
        int[] meta = new int[9];
        long mesh = GeoModel.nBuildGpuMesh(gpuWire(bones), meta);
        gpuAlive = mesh != 0;
        out.println("gpu.handle!=" + gpuAlive);
        out.println("gpu.meta=" + hex(meta, 9));
        out.flush();
        if (mesh != 0) {
            ByteBuffer vb = GeoModel.nGetGpuMeshVertexBuffer(mesh);
            ByteBuffer ib = GeoModel.nGetGpuMeshIndexBuffer(mesh);
            int vCount = meta[0];
            int iCount = meta[1];
            out.println("gpu.v=" + hex(vb, vCount * 32));
            out.println("gpu.i=" + hex(ib, iCount * 4));

            ByteBuffer outBone = ByteBuffer.allocateDirect(boneCount * 144)
                    .order(ByteOrder.nativeOrder());
            GeoModel.nComputeBoneMatricesLocal(mesh, anim, light, outBone);
            out.println("gpu.boneOut=" + hex(outBone, boneCount * 144));
            // world-mode matrices too (nComputeBoneMatrices: rootPose/rootNormal arrays)
            float[] pose = new float[16], normal = new float[16];
            System.arraycopy(mats, 0, pose, 0, 16);
            System.arraycopy(mats, 16, normal, 0, 16);
            GeoModel.nComputeBoneMatrices(mesh, pose, normal, anim, light, outBone);
            out.println("gpu.boneOutWorld=" + hex(outBone, boneCount * 144));
            gpuHidden = new int[boneCount];
            for (int i = 0; i < boneCount; i++) gpuHidden[i] = isHiddenAt(outBone, i);
            out.println("gpu.isHidden=" + java.util.Arrays.toString(gpuHidden));

            GeoModel.nReleaseGpuMeshScratch(mesh);
            GeoModel.nFreeGpuMesh(mesh);
        }
        out.flush();

        GeoModel.nDestroyModelCache(handle);
        out.flush();

        if (semantic) {
            int rc = semanticCheck(testCase, boneCount, out);
            out.close();
            System.exit(rc);
        }
        out.close();
    }

    /**
     * Real assertions (exit 1 on mismatch) for the offset9 / non-tree fixes.
     * Expected values derived from CPU-path parity (calculateBoneMatrix):
     * offset9!=0 -> bone invisible, subtree invisible; non-tree inputs must
     * not crash and must render every bone.
     */
    private static int semanticCheck(String testCase, int boneCount, PrintWriter out) {
        String fail = null;
        switch (testCase) {
            case "offset9Self" -> {
                if (slowCount != 4 || fastVertices != 4) fail = "slow=" + slowCount + " fast=" + fastVertices + " want 4/4";
                if (fail == null && !java.util.Arrays.equals(gpuHidden, new int[]{0, 1}))
                    fail = "hidden=" + java.util.Arrays.toString(gpuHidden) + " want [0, 1]";
            }
            case "offset9Root" -> {
                if (slowCount != 0 || fastVertices != 0) fail = "slow=" + slowCount + " fast=" + fastVertices + " want 0/0";
                if (fail == null && !java.util.Arrays.equals(gpuHidden, new int[]{1, 1, 1}))
                    fail = "hidden=" + java.util.Arrays.toString(gpuHidden) + " want [1, 1, 1]";
            }
            case "offset9Midtree" -> {
                if (slowCount != 8 || fastVertices != 8) fail = "slow=" + slowCount + " fast=" + fastVertices + " want 8/8";
                if (fail == null && !java.util.Arrays.equals(gpuHidden, new int[]{0, 1, 1, 0}))
                    fail = "hidden=" + java.util.Arrays.toString(gpuHidden) + " want [0, 1, 1, 0]";
            }
            case "offset9PlusSkip10" -> {
                if (slowCount != 0 || fastVertices != 0) fail = "slow=" + slowCount + " fast=" + fastVertices + " want 0/0";
                if (fail == null && !java.util.Arrays.equals(gpuHidden, new int[]{1, 1}))
                    fail = "hidden=" + java.util.Arrays.toString(gpuHidden) + " want [1, 1]";
            }
            case "offset9GpuOnly" -> {
                if (slowCount != 8 || fastVertices != 8) fail = "slow=" + slowCount + " fast=" + fastVertices + " want 8/8";
                if (fail == null && !java.util.Arrays.equals(gpuHidden, new int[]{0, 1, 1}))
                    fail = "hidden=" + java.util.Arrays.toString(gpuHidden) + " want [0, 1, 1]";
            }
            case "nontreeBadParent", "nontreeCycle" -> {
                if (!cacheAlive) fail = "cache handle=0";
                if (fail == null && !gpuAlive) fail = "gpu handle=0";
                if (fail == null && (slowCount != 8 || fastVertices != 8))
                    fail = "slow=" + slowCount + " fast=" + fastVertices + " want 8/8";
            }
            default -> {
                out.println("semantic.case=" + testCase);
                out.println("semantic.result=SKIP (no assertions)");
                return 0;
            }
        }
        out.println("semantic.case=" + testCase);
        out.println("semantic.result=" + (fail == null ? "PASS" : "FAIL " + fail));
        return fail == null ? 0 : 1;
    }
}
