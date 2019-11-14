(define (g x) (* x x))
(define (f x y) 
    (set! g 5)
    x
)