;                    0 1       2 3 4
(define (f? x y) (a? x (= x y) y y (= x y)))
;           0 1  2 3  4
(define (a? a b? c d? e) (= a b))
; ;                    0 1       2 3 4
; (define (f? x y) (a? (= x y) (= x y) (= x y) (= x y) (= x y)))
; ;           0 1  2 3  4
; (define (a? x? b? c? d? e?) (= a b))