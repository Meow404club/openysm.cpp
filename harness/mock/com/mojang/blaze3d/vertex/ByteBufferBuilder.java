package com.mojang.blaze3d.vertex;

/**
 * Harness double of the 26.2 blaze3d ByteBufferBuilder, compiled under its
 * REAL FQN: the native staging probe resolves the BufferBuilder.buffer field
 * by the descriptor "Lcom/mojang/blaze3d/vertex/ByteBufferBuilder;", so the
 * mock must shadow that exact name (same trick as the shaded GeoModel stub).
 * vanilla 26.2 ByteBufferBuilder.java:49-55: reserve(int) advances writeOffset,
 * grows the allocation and returns the ABSOLUTE write address pointer+offset.
 *
 * Storage is raw Unsafe memory: the staging fast path writes at the raw
 * jlong reserve() returns, so the mock's backing store must be a real native
 * address (a direct ByteBuffer object has no Java-readable address).
 */
public class ByteBufferBuilder {
    private static final sun.misc.Unsafe UNSAFE;
    static {
        try {
            java.lang.reflect.Field f = sun.misc.Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            UNSAFE = (sun.misc.Unsafe) f.get(null);
        } catch (Exception e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private long pointer;
    private long capacity;
    private long writeOffset;

    public int reserveCalls;
    public int lastReserveArg = -1;

    public ByteBufferBuilder(int initialCapacity) {
        pointer = UNSAFE.allocateMemory(initialCapacity);
        capacity = initialCapacity;
    }

    public long reserve(int size) {
        reserveCalls++;
        lastReserveArg = size;
        long nextOffset = writeOffset + size;
        ensureCapacity(nextOffset);
        long addr = pointer + writeOffset;
        writeOffset = nextOffset;
        return addr;
    }

    private void ensureCapacity(long required) {
        if (required > capacity) {
            long newCapacity = Math.max(required, capacity * 2);
            pointer = UNSAFE.reallocateMemory(pointer, newCapacity);
            capacity = newCapacity;
        }
    }

    /** Snapshot of everything reserve() has handed out so far. */
    public byte[] writtenBytes() {
        byte[] out = new byte[(int) writeOffset];
        for (int i = 0; i < out.length; i++) out[i] = UNSAFE.getByte(pointer + i);
        return out;
    }

    public long writeOffset() {
        return writeOffset;
    }
}
