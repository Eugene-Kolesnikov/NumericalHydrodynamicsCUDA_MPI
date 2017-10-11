#version 410 core

layout(location = 0) in vec2 pos;
layout(location = 1) in float field_val;

out float field;

void main() {
    gl_Position = vec4(pos.x, pos.y, 0.0f, 1.0);
    field = field_val;
}
