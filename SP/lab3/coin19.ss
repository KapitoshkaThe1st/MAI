(define VARIANT 9)
(define LAST-DIGIT-OF-GROUP-NUMBER 8)
(define LARGEST-COIN 20)

(define (implication? x? y?) (not (and x? (not y?))))

(define (cc amount largest-coin) (cond ((or (= amount 0) (= largest-coin 1)) 1)
                                 ((implication? (>= amount 0) (= largest-coin 0)) 0)
                                 (else (+ (cc amount (next-coin largest-coin)) 
                                    (cc (- amount largest-coin) largest-coin))) )
)

(define (count-change amount) (cc amount LARGEST-COIN))

(define (next-coin coin) (cond ((= coin 20) 15)
                        ((= coin 15) 10)
                        ((= coin 10) 5)
                        ((= coin 5) 3)
                        ((= coin 3) 2)
                        (else 1) )
)

(define (GR-AMOUNT) (remainder (+ (* 100 LAST-DIGIT-OF-GROUP-NUMBER) VARIANT) 137))

(display " KAV variant ")
(display VARIANT) (newline)
(display " 1-2-3-5-10-15-20") (newline)
(display "count__change for 100 \t= ")
(display (count-change 100)) (newline)
(display "count__change for ");
(display (GR-AMOUNT))
(display " \t= ")
(display (count-change (GR-AMOUNT)))(newline)