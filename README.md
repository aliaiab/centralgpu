# centralgpu

A gpu-like software rasterizer.

# Basic Design

## Geometry processing 

The triangle pipeline uses 8-wide simd, with each lane processing a triangle.
Per vertex processing is done in lockstep, processing each vertex one after the other.

Having each lane processing a triangle rather than a vertex allows maximal utilization of the simd width,
and triangles don't have to be "assembled" at any point. The rasterizer simply takes in 8 triangles and iterates.

## Rasterization/Fragment shading

The rasterizer takes in 8 projected triangles directly from geometric processing.
Triangle setup is done 8-wide just as is done in geometric processing.

The rasterizer itself shades pixels in pairs of 2x2 'quads'. This allows for partial derivatives to 
be computed which is essential for doing mipmapping.

