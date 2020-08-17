(defun aux (g a x) 
    (if x (aux g (funcall g a (car x)) (cdr x)) a)
)

(defun редукция2 (g a x)
    (if x (aux g a x) nil)
)

; (редукция2 #'+ 0 '(1 2 3 4 5))
; (редукция2 #'expt 5 '(1 2 5))
; (редукция2 (lambda (у z) (+ (* 10 у) z)) 0 '(1 3 4 5))
