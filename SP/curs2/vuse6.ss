(define (f x y) (* x y))

(define (c t) 
    (let ((q 5) (r 7))
        (* f r t)
    )
)