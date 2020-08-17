(defclass cart ()
 ((x :initarg :x :reader x)
  (y :initarg :y :reader y)))

(defclass polar ()
 ((rad :initarg :rad :accessor rad)
  (angle  :initarg :angle  :accessor angle)))

(defclass line ()
 ((start :initarg :start :accessor start)
  (end   :initarg :end   :accessor end)))
     

(defun polar-to-decart (r phi)
    (list (* r (cos phi)) (* r (sin phi)) )
)

(defvar eps 0.0001)
(defun approx-equal (a b)
    (< (abs (- a b)) eps)
)

(defun f (k line)
    (let* (
            (start (start line))
            (end (end line))
            (p1 (if (eq (type-of start) 'polar) 
                (polar-to-decart (rad start) (angle start) )
                (list (x start) (y start) ))
            )
            (p2 (if (eq (type-of end) 'polar) 
                (polar-to-decart (rad end) (angle end) )
                (list (x end) (y end) ))
            )
            (k1 
                (/ (- (nth 1 p2) (nth 1 p1)) (- (nth 0 p2) (nth 0 p1)))
            )
        )
        (approx-equal k1 k)
    )
)
     
(defun aux (k l)
    (if l 
        (and (f k (car l)) (aux k (cdr l)))
        t
    )
)

(defun line-parallel-p (lines)
        (let* (
            (line (car lines))
            (start (start line))
            (end (end line))
            (p1 (if (eq (type-of start) 'polar) 
                (polar-to-decart (rad start) (angle start) )
                (list (x start) (y start) ))
            )
            (p2 (if (eq (type-of end) 'polar) 
                (polar-to-decart (rad end) (angle end) )
                (list (x end) (y end) ))
            )
            (k 
                (/ (- (nth 1 p2) (nth 1 p1)) (- (nth 0 p2) (nth 0 p1)))
            )
        )
        (aux k (cdr lines))
    )
)


                  

