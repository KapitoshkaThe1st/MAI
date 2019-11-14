;even-odd

(define var1 1)
(define var2 2)
(define var1000000 1000000)
(define var10000 10000)


(define(even-bits n)
  (cond((= n 0)var1)
       ((=(remainder n var2)0)
          (even-bits (quotient n var2)))
       (#t (odd-bits(quotient n var2)))
       ))
(define(odd-bits n)
  (cond((= n 0)0)
       ((=(remainder n var2)0)
          (odd-bits (quotient n var2)))
       (#t (even-bits(quotient n var2)))
       ))
(define(display-bin n)
  (display(remainder n var2))
  (cond ((= n 0) 0) (#t(display-bin (quotient n var2))))
       )
(define(report-results n)
  (display "Happy birthday to you!\n\t")
  (display n)(display " (decimal)\n\t")
  (display-bin n)(display "(reversed binary)\n")
  (display "\teven?\t")(display (if(=(even-bits n)var1) "yes" "no"))
  (newline)
  (display "\todd?\t")(display (if(=(odd-bits n)var1) "yes" "no"))
  (newline)
  0
       )
;***** Date of YOUR birthday *******
(define dd 29)
(define mm 4)
(define yyyy 1999)
;***********************************
(report-results (+ (* dd var1000000)
                   (* mm var10000)
                   yyyy))


 
