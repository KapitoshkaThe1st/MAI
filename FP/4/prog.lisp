(defun prior (op)
    (cond 
        ((eq op '+) 1)
        ((eq op '-) 2)
        ((eq op '*) 3) 
        ((eq op '/) 4)
        (t 0)
    )
)

(defun non-associative (op)
    (or (eq op '-) (eq op '/))
)

(declaim (ftype (function (t t) t) g))

(defun f (op l)
    (if (cdr l) 
        (concatenate 'string (g (car l) op) " " (write-to-string op) " " (f op (cdr l)))
        (g (car l) op)
    )
)

(defun h (l external-op)
    (cond 
        ((= (list-length l) 2)
            (cond
                ((eq (car l) '-)
                    (concatenate 'string "(- " (f (car l) (cdr l)) ")"))
                ((eq (car l) '/)
                    (concatenate 'string "1 / " (f (car l) (cdr l))))
                (t 
                    (f (car l) (cdr l)) )
            )
             
        )
        ((> (prior external-op) (prior (car l)))
            (concatenate 'string "(" (f (car l) (cdr l)) ")")    
        )
        ((= (prior external-op) (prior (car l)))
            (if (non-associative external-op)
                (concatenate 'string "(" (f (car l) (cdr l)) ")")
                (f (car l) (cdr l))
            )
        )
        (t (f (car l) (cdr l)))
    )
)

(defun g (l external-op) 
    (if (atom l) 
        (write-to-string l)
        (h l external-op)
    )
)

(defun form-to-infix (l)
    (h l '+)
)

; (trace f)
; (trace g)
; (trace h)
; (trace prior)

;; (print (form-to-infix '(+ (* b b) (- (* 4 a c)))) )
;; (print (form-to-infix '(* (* a b) (* c (* d e)))))
;; (print (form-to-infix '(/ (- b c d) a) ))

;; (print (form-to-infix '(- (+ A (- (* B C))) D) ))
;; (print (form-to-infix '(* (+ A B) (+ C D)) ))
;; (print (form-to-infix '(- (+ (+ A B) C) D) ))
;; (print (form-to-infix '(+ (* A B) (* C D)) ))
;; (print (form-to-infix '(* (/ a) c) ))

