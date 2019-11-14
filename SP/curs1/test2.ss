;golden-section

(define var5 5)
(define var2 2)
(define var3 3)

(define (f a b c)
    (cond 
        ((< a b) a)
        ((< a c) a)
    )
)
(maximum var5 var2 var3)