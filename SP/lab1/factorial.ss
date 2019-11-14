(define(factorial n)
  (if(< n 2)
     1
     (* n (factorial (- n 1)))))
"Factorials:"
(factorial 5)
(factorial 10)

