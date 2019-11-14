;lisp05
(define(f x)(* 2 x))
(f 3)
;________________________
(define(g x y)(+ (f x) y))
(g 4 5)
;________________________
(define(dx) 1e-5)
(dx)

