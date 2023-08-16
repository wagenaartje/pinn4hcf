//+
Point(1) = {0.5, -0, 0, 1.0};
//+
Point(2) = {-0, -0, 0, 1.0};
//+
Point(3) = {0, 4.0, 0, 1.0};
//+
Point(4) = {2.0, 4.0, 0, 1.0};
//+
Point(5) = {2.0, 0.25, 0, 1.0};//+
Point(6) = {1.0, 0.25, 0, .0};
//+
Bezier(1) = {1, 6, 5};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 3};
//+
Line(5) = {4, 5};
//+
Point(7) = {0.5, 4.0, 0, 1.0};
//+
Line(6) = {3, 7};
//+
Line(7) = {7, 4};
//+
Line(8) = {7, 1};
//+
Transfinite Line {3} = 200 Using Progression 1.01;
//+
Transfinite Line {8,5} = 200 Using Progression 0.99;
//+
Transfinite Line {2,6} = 80 Using Progression 1;
//+
Transfinite Line {1,7} = 320 Using Progression 1;//+
Curve Loop(1) = {3, 6, 8, 2};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {8, 1, -5, -7};
//+
Plane Surface(2) = {2};
//+
Transfinite Surface {1};
//+
Transfinite Surface {2};
//+
Recombine Surface {1, 2};
//+
Extrude {0, 0, 1} {
  Surface{1}; Surface{2}; Layers {1}; Recombine;
}
//+
Physical Volume("Fluid", 53) = {1, 2};
//+
Physical Surface("inlet", 54) = {17};
//+
Physical Surface("top", 55) = {21, 51};
//+
Physical Surface("bottom", 56) = {29};
//+
Physical Surface("obstacle", 57) = {43};
//+
Physical Surface("outlet", 58) = {47};
