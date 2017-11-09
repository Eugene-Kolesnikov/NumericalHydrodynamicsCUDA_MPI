# Dynamic Library of Visualization (DLV)

In this particular case, DLV uses OpenGL to render the field and saves it to one of the following files: ppm, png, mpeg (for videos). Since the software solves 2D problems, the Visualization is a 2D image (static or dynamic depending on the problem), which is generated using OpenGL 4.x and GLFW 3 as an interface generation library.

Interface functions:

```
bool DLV_init(size_t N_X, size_t N_Y, enum OUTPUT_OPTION outOption);
bool DLV_visualize(void* field, size_t N_X, size_t N_Y);
bool DLV_terminate();
```

The field should be normalized to [0,1] before sending it to the ```DLV_visualize``` function. The DLV renders the field on a square $[0,1]^2$.
\\[ DLV: [0,1] -> [0,1]^3 \\]



# TODO:

1. Update the documentation
2. Rewrite the visualization library using Qt and add functionality to save raw data of the simulation for further rerendering
3. Create a tool for rerendering from existing raw data