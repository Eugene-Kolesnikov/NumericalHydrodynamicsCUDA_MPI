#version 410 core

in float field;
out vec4 fColor;

vec3 colorbar(float field)
{
    vec3 color = vec3(0,0,0);
    /*if(field >= 2.0/3.0) {
        float d = field - 2.0/3.0;
        color = vec3(0,(1.0/3.0-d)*3.0,d*3.0);
    } else {
        if(field >= 1.0/3.0) {
            float d = field - 1.0/3.0;
            color = vec3((1.0/3.0-d)*3.0,d*3.0,0);
        } else {
            color = vec3(field*3.0,0,0);
        }
    }*/
    if(field >= 1.0/2.0) {
        float d = field - 1.0/2.0;
        color = vec3(0,(1.0/2.0-d)*2.0,d*2.0);
    } else {
        float d = 1.0/2.0 - field;
        color = vec3(d*2.0,(1.0/2.0-d)*2.0,0);  
    }
    return color;
}

void main (void) {
    
    fColor = vec4(colorbar(field),1);
}
