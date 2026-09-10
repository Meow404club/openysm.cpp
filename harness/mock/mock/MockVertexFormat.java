package mock;

/**
 * Mock of VertexFormat + its Mode enum. The native side resolves the Mode
 * field descriptor at runtime via Class.getName(), so only the class shape
 * matters, not the actual name.
 */
public class MockVertexFormat {
    public enum Mode {
        QUADS,
        TRIANGLE_STRIP,
        TRIANGLES
    }
}
