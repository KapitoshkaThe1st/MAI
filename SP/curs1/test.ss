;golden-section

(define var5 5)
(define var2 2)
(define var3 3)

(define (maximum a b c)
    (cond 
        ((< a b) (display "huy") var2)
        ((= a b) (display "huy") (display "pizda") var3)
        ((< a b) (display "huy") var2) 
        (#t var5 )
    )
)
(maximum var5 var2 var3)