int: n;
int: m;
array[1..m, 1..2] of int: E;
int: maxK = n;

array[1..n] of var 1..maxK: c;

constraint forall(i in 1..m)( c[E[i, 1]] != c[E[i, 2]] );

var 1..maxK: maxColor = max(c);

solve :: int_search(c, first_fail, indomain_min, complete) minimize maxColor;
