package mock;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Mock of the vanilla BufferBuilder fast-write target. Field and method names
 * match what GeoModel.nInitSIMD binds at runtime:
 *   buffer           : Ljava/nio/ByteBuffer;   (direct buffer)
 *   vertices         : I
 *   nextElementByte  : I
 *   ensureCapacity(I): V
 *   mode             : Lmock/MockVertexFormat$Mode;
 */
public class MockBufferBuilder {
    public ByteBuffer buffer;
    public int vertices;
    public int nextElementByte;
    public MockVertexFormat.Mode mode = MockVertexFormat.Mode.QUADS;

    public int ensureCapacityCalls = 0;
    public int lastEnsureCapacityArg = -1;

    public MockBufferBuilder(int capacity) {
        buffer = ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    @SuppressWarnings("unused")
    public void ensureCapacity(int size) {
        ensureCapacityCalls++;
        lastEnsureCapacityArg = size;
        if (buffer.capacity() < size) {
            ByteBuffer nb = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder());
            buffer = nb;
        }
    }
}
