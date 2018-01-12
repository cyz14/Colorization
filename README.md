# Colorization Using Optimization

http://www.cs.huji.ac.il/~yweiss/Colorization/index.html

## Notes

1. Bmp files cannot be too large.

Scipy will fail when sparse matrix is too large on windows.
Other platforms still need tests.

2. Stroke should not have anti-aliasing

When you use Photoshop or any other image editors to add scribbles,
use solid stroke without anti-alising effect, or color will not expand around.
In photoshop I use pencil at last.
