; t6
(define(try x)(set! x(f x x))x)
(define(f x y)(* x y))
(try 3)
