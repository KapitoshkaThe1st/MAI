(define (t a b) a)
(define (f x y)
    (let ((t 6) (r 7)) 
        (set! t 5)
        x
    )
)