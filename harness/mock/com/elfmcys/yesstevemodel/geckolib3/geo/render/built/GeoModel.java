package com.elfmcys.yesstevemodel.geckolib3.geo.render.built;

import java.nio.ByteBuffer;

/**
 * Harness stub of the real GeoModel native-binding class. The .so binds via
 * RegisterNatives on this exact FQN (see JNI_OnLoad), so the package and name
 * must not change. Mirrors the native declarations of
 * common/src/main/java/.../GeoModel.java:198-250.
 */
public class GeoModel {
    static {
        String lib = System.getenv("YSM_LIB");
        if (lib == null || lib.isEmpty()) {
            throw new IllegalStateException("YSM_LIB not set");
        }
        System.load(lib);
    }

    public static native void nInitSIMD(
            Class<?> bufferBuilderClass,
            String bufferName,
            String verticesName,
            String nextElementByteName,
            String ensureCapacityName,
            String modeName,
            Class<?> vertexFormatModeClass);

    public static native long nInitModelCache(ByteBuffer buffer);

    public static native void nDestroyModelCache(long handle);

    public static native void nComputeModelVertices(
            long handle,
            Object vertexConsumer,
            float[] matrixArray,
            float[] animArray,
            float[] stateArray,
            int renderPartMask,
            int packedLight,
            int packedOverlay,
            float r, float g, float b, float a);

    public static native long nBuildGpuMesh(ByteBuffer buffer, int[] outMeta);

    public static native ByteBuffer nGetGpuMeshVertexBuffer(long pointer);

    public static native ByteBuffer nGetGpuMeshIndexBuffer(long pointer);

    public static native void nReleaseGpuMeshScratch(long pointer);

    public static native void nFreeGpuMesh(long pointer);

    public static native void nComputeBoneMatrices(long pointer, float[] rootPose,
                                                   float[] rootNormal, float[] anim,
                                                   int packedLight, ByteBuffer outBoneBuffer);

    public static native void nComputeBoneMatricesLocal(long handle, float[] animArray,
                                                        int packedLight, ByteBuffer outBoneBuffer);
}
