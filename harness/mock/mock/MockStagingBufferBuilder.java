package mock;

import com.mojang.blaze3d.vertex.ByteBufferBuilder;

/**
 * Mock of the 26.2 BufferBuilder field face (neoform 26.2 BufferBuilder.java:
 * 21-31, diffed field-identical against 26.3):
 *   buffer      : Lcom/mojang/blaze3d/vertex/ByteBufferBuilder; (NOT ByteBuffer)
 *   vertices    : I
 *   vertexSize  : I (final in vanilla; 36 for DefaultVertexFormat.ENTITY)
 *   nextElementByte / ensureCapacity / mode : GONE (probes must miss).
 */
public class MockStagingBufferBuilder {
    public ByteBufferBuilder buffer;
    public int vertices;
    public int vertexSize = 36;

    public MockStagingBufferBuilder(int capacity) {
        buffer = new ByteBufferBuilder(capacity);
    }
}
