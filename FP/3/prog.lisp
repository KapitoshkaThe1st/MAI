(defun aux (n)
    (let (
        (a (make-array (list n n)))
        (k 1)
        )
        (dotimes (m (- (* 2 n) 1) )
            (if (= (mod m 2) 1)
                (let* (
                    (q (max 0 (+ (- m n) 1) ))
                    (i (min (- n 1) m))
                    (j q) )
                    (loop while (>= i q) do
                        (setf (aref a i j) k)
                        (setq k (+ k 1))
                        (setq i (- i 1))
                        (setq j (+ j 1))
                    )
                )
                (let* (
                    (q (max 0 (+ (- m n) 1) ))
                    (i q)
                    (j (min (- n 1) m)) )
                    (loop while (>= j q) do
                        (setf (aref a i j) k)
                        (setq k (+ k 1))
                        (setq i (+ i 1))
                        (setq j (- j 1))
                    )
                )
            )
        )
        a
    )
)

(defun matrix-1l-2l (n)
    (if (> n 0) (aux n) nil)
)

;; (matrix-1l-2l 1)
;; (matrix-1l-2l 5)
;; (matrix-1l-2l -13)