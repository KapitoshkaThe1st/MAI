;lisp04
(define a 1)
(define b 2)
(cond ((< a b) 1)         ;#1
      ((= a b) 0)
      ((> a b) 1) )
(cond ((= a b) 0)          ;#2
      (else 1)     )
(if(= a b) 0 1)            ;#3
(if(or(< a b)              ;#4
      (> a b) ) 1 0)
(if(and(not(< a b))        ;#5
       (not(> a b)) ) 0 1)
