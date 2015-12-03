#version 330 core
//http://outerra.blogspot.ca/2013/07/logarithmic-depth-buffer-optimizations.html
//fix same bugs with z-buffer and model killeroo
out vec4 glcolor;
uniform vec4 color;
in float flogz;
const float far = 1000.0f;
const float Fcoef = 2.0 / log2(far + 1.0);
const float Fcoef_half = 0.5 * Fcoef;
void main() {
    glcolor = color;
    gl_FragDepth = log2(flogz) * Fcoef_half;
}
