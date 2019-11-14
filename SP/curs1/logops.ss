(define (my-not? x?) 
  (= 0 (cond (x? e) (#t 0)) )
)

(define (my-and? x? y?)
  (= e (cond (x? e) (#t 0)) (cond (y? e) (#t 0)) )
)

(define (my-or? x? y?)
  (= e (cond (x? e) (#t (cond (y? e) (#t 0))) ) )
)

(define (implication? x? y?)
  (my-or? (my-not? x?) y?)
)

(my-not? #t)
(newline)

(my-and? #f #f)
(my-and? #f #t)
(my-and? #t #f)
(my-and? #t #t)
(newline)

(my-or? #f #f)
(my-or? #f #t)
(my-or? #t #f)
(my-or? #t #t)
(newline)

(implication? #f #f)
(implication? #f #t)
(implication? #t #f)
(implication? #t #t)
(newline)