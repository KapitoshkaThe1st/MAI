; функция четности числа
(defun parity (x) (rem x 2))

; за один шаг из клетки (k, l) в клетку (m, n) можно добраться, если равны модули разниц соответствующих координат
(defun one-step (k l m n)
    (= (abs (- k m)) (abs (- l n)))
)

; проверяем рекурсивно / диагональ
(defun aux1 (k l m n) 
    (cond 
        ((or (= k 9) (= l 9)) nil)
        ((one-step k l m n) (list k l))
        (T (aux1 (+ k 1) (+ l 1) m n))
    )
)

; проверяем рекурсивно \ диагональ
(defun aux2 (k l m n) 
    (cond 
        ((or (= k 0) (= l 9)) nil)
        ((one-step k l m n) (list k l))
        (T (aux2 (- k 1) (+ l 1) m n))
    )
)

; проверяем по очереди элементы но двух диагональных линиях, проходящих через точку (k, l)
(defun two-steps (k l m n)
    (let* ((d1 (- (min k l) 1))
          (d2 (min (- 8 k) l))
          (res (aux1 (- k d1) (- l d1) m n)))
        (if res res (aux2 (+ k d2) (- l d2) m n))
    )
)

; неравенство четности сумм коорданат точек означает, что эти клетки разного цвета, и слон не может попасть туда совсем.
(defun bishop-moves (k l m n) 
    (cond 
        ((one-step k l m n) T)
        ((/= (parity (+ k l)) (parity (+ m n)) ) nil)
        (T (values-list (two-steps k l m n)))
    )
)