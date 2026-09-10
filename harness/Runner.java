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
 * Usage: java Runner <case> <outFile>
 * env:   YSM_LIB = absolute path of the libysm-core.so under test.
 *
 * Builds the model wire buffer exactly like GeoModel.buildNativeCache()
 * (4 + bones*25 + cubes*5 + quads*93 bytes; per quad: translucent byte,
 * 12 float positions, 8 float uvs, 3 float normal), drives the JNI surface
 * through both the submitVertices callback path (vertexConsumer = null) and
 * the nInitSIMD BufferBuilder direct-write path, plus the GPU mesh path, and
 * dumps every observable output as hex lines for byte-level diffing.
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
            case "gpuMesh" -> {
                // Same shape as "translucent": drives nBuildGpuMesh +
                // nComputeBoneMatricesLocal (glow bone included).
                bones.add(new Bone(-1, 1, true, 1, 2, 3, List.of(
                        new Cube(true, List.of(quad(false, true, 5f, 0f))),
                        new Cube(true, List.of(quad(true, true, 5f, 0.25f))))));
                bones.add(new Bone(0, 2, false, 4, 2, 0, List.of(
                        new Cube(false, List.of(quad(true, false, 6f, 0.5f))))));
            }
            default -> throw new IllegalArgumentException("unknown case " + testCase);
        }
        return bones;
    }

    private static float[] animFor(String testCase, int boneCount) {
        float[] anim = new float[boneCount * 12];
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
            anim[o + 9] = 0f;                // unk9
            anim[o + 10] = "hiddenSubtree".equals(testCase) && b == 1 ? 1f : 0f; // skip flag
            anim[o + 11] = "state".equals(testCase) ? 1f : 0f;                   // state sentinel
        }
        if ("zeroScale".equals(testCase)) {
            anim[12 + 6] = 0f;
            anim[12 + 7] = 0f;
            anim[12 + 8] = 0f;
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

    // ----- wire buffer (mirrors GeoModel.buildNativeCache) ------------------

    private static ByteBuffer wire(List<Bone> bones) {
        int quads = 0;
        for (Bone b : bones) for (Cube c : b.cubes()) quads += c.quads().size();
        int size = 4 + bones.size() * 25 + quads * 0; // cubes counted below
        int cubes = 0;
        for (Bone b : bones) cubes += b.cubes().size();
        size = 4 + bones.size() * 25 + cubes * 5 + quads * 93;
        ByteBuffer buf = ByteBuffer.allocate(size).order(ByteOrder.nativeOrder());
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
                    buf.put((byte) (q.translucent() ? 1 : 0));
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

    // ----- driver -------------------------------------------------------------

    public static void main(String[] args) throws Exception {
        String testCase = args[0];
        String outPath = args[1];
        PrintWriter out = new PrintWriter(outPath, "UTF-8");

        List<Bone> bones = bonesFor(testCase);
        int boneCount = bones.size();
        float[] anim = animFor(testCase, boneCount);

        // Register fast-path handles against the mocks.
        GeoModel.nInitSIMD(MockBufferBuilder.class, "buffer", "vertices", "nextElementByte",
                "ensureCapacity", "mode", MockVertexFormat.Mode.class);

        long handle = GeoModel.nInitModelCache(wire(bones));
        out.println("handle!=" + (handle != 0));
        if (handle == 0) {
            out.close();
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
        out.println("slow.calls=" + NativeModelRenderer.callCount);
        out.println("slow.count=" + NativeModelRenderer.lastVertexCount);
        if (NativeModelRenderer.lastFBuf != null && NativeModelRenderer.lastVertexCount > 0) {
            int fLen = NativeModelRenderer.lastVertexCount * 12 * 4;
            int iLen = NativeModelRenderer.lastVertexCount * 2 * 4;
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
        out.println("fast.ensureCalls=" + builder.ensureCapacityCalls);
        out.println("fast.ensureArg=" + builder.lastEnsureCapacityArg);
        out.println("fast.written=" + builder.nextElementByte);
        out.println("fast.buffer=" + hex(builder.buffer, Math.max(0, builder.nextElementByte)));
        out.println("fast.vertices=" + builder.vertices);

        // --- partMask pass (slow only; mask=2) ---
        NativeModelRenderer.callCount = 0;
        GeoModel.nComputeModelVertices(handle, null, mats, anim, state,
                2, light, overlay, 1f, 1f, 1f, 1f);
        out.println("mask.calls=" + NativeModelRenderer.callCount);
        out.println("mask.count=" + NativeModelRenderer.lastVertexCount);
        if (NativeModelRenderer.lastVertexCount > 0) {
            out.println("mask.f=" + hex(NativeModelRenderer.lastFBuf,
                    NativeModelRenderer.lastVertexCount * 12 * 4));
        }

        // --- GPU mesh path ---
        int[] meta = new int[9];
        long mesh = GeoModel.nBuildGpuMesh(wire(bones), meta);
        out.println("gpu.handle!=" + (mesh != 0));
        out.println("gpu.meta=" + hex(meta, 9));
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

            GeoModel.nReleaseGpuMeshScratch(mesh);
            GeoModel.nFreeGpuMesh(mesh);
        }

        GeoModel.nDestroyModelCache(handle);
        out.close();
    }
}
