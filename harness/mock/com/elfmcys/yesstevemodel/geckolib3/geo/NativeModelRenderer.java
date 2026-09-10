package com.elfmcys.yesstevemodel.geckolib3.geo;

import java.nio.ByteBuffer;

/**
 * Harness stub of NativeModelRenderer: the .so looks up the static method
 * submitVertices(Object,int,ByteBuffer,ByteBuffer) on this exact FQN inside
 * JNI_OnLoad. Records the callback arguments for byte-level comparison.
 */
public class NativeModelRenderer {
    public static Object lastVertexConsumer;
    public static int lastVertexCount = -1;
    public static ByteBuffer lastFBuf;
    public static ByteBuffer lastIBuf;
    public static int callCount = 0;

    @SuppressWarnings("unused")
    public static void submitVertices(Object vertexConsumer, int vertexCount,
                                      ByteBuffer fBuf, ByteBuffer iBuf) {
        lastVertexConsumer = vertexConsumer;
        lastVertexCount = vertexCount;
        lastFBuf = fBuf;
        lastIBuf = iBuf;
        callCount++;
    }
}
