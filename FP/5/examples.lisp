(defvar lines (list (make-instance 'line
                   :start (make-instance 'cart :x 3 :y 5)
                   :end (make-instance 'cart :x 7 :y 10)
                  )
                  (make-instance 'line
                  :start (make-instance 'cart :x 3 :y 6)
                  :end (make-instance 'cart :x 7 :y 11)
                  )
                  (make-instance 'line
                  :start (make-instance 'polar :rad 2.25 :angle (/ pi 2))
                  :end (make-instance 'cart :x 7 :y 11)
                  )
                  (make-instance 'line
                  :start (make-instance 'polar :rad 13.9283882772 :angle 1.203622493)
                  :end (make-instance 'polar :rad 31.169897337 :angle 1.0317523415)
                  )
              )
)

(print (line-parallel-p lines))
    
    
(defvar lines (list (make-instance 'line
                   :start (make-instance 'cart :x 3 :y 5)
                   :end (make-instance 'cart :x 7 :y 10)
                  )
                  (make-instance 'line
                  :start (make-instance 'cart :x 3 :y 6)
                  :end (make-instance 'cart :x 7 :y 11)
                  )
                  (make-instance 'line
                  :start (make-instance 'polar :rad 2.25 :angle (/ pi 2))
                  :end (make-instance 'cart :x 7 :y 11)
                  )
                  (make-instance 'line
                  :start (make-instance 'polar :rad 13.9513882772 :angle 1.203622493) ;; отличие от соответствующего числа выше в двух знаках, начиная со 2-го после запятой
                  :end (make-instance 'polar :rad 31.169897337 :angle 1.0317523415)
                  )
              )
) 

(print (line-parallel-p lines))
