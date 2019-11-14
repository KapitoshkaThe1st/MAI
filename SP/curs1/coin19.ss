(define VARIANT 9)
(define LAST-DIGIT-OF-GROUP-NUMBER 8)
(define LARGEST-COIN 20)

(define var1 1)
(define var2 2)
(define var3 3)
(define var5 5)
(define var10 10)
(define var15 15)
(define var20 20)
(define var100 100)
(define var137 137)

(define (my-not? x?) 
  (= 0 (cond (x? e) (#t 0)) )
)

(define (my-or? x? y?)
  (= e (cond (x? e) (#t (cond (y? e) (#t 0))) ) )
)

(define (implication? x? y?)
  (my-or? (my-not? x?) y?)
)

(define (cc amount largest-coin) (cond ((my-or? (= amount 0) (= largest-coin var1)) var1)
                                 ((implication? (my-not? (< amount 0)) (= largest-coin 0)) 0)
                                 (#t (+ (cc amount (next-coin largest-coin)) 
                                    (cc (- amount largest-coin) largest-coin))) )
)

(define (count-change amount) (cc amount LARGEST-COIN))

(define (next-coin coin) (cond ((= coin var20) var15)
                        ((= coin var15) var10)
                        ((= coin var10) var5)
                        ((= coin var5) var3)
                        ((= coin var3) var2)
                        (#t var1) )
)

(define (GR-AMOUNT) (remainder (+ (* var100 LAST-DIGIT-OF-GROUP-NUMBER) VARIANT) var137))

(display " KAV variant ")
(display VARIANT) (newline)
(display " 1-2-3-5-10-15-20") (newline)
(display "count__change for 100 \t= ")
(display (count-change var100)) (newline)
(display "count__change for ");
(display (GR-AMOUNT))
(display " \t= ")
(display (count-change (GR-AMOUNT)))(newline)