/* $mlisp19 */
int tSM::p02(){ //    PROG -> CALCS
	return 0;}
int tSM::p03(){ //    PROG -> DEFS
	return 0;}
int tSM::p04(){ //    PROG -> DEFS CALCS
	return 0;}
int tSM::p05(){ //   CALCS -> CALC
	return 0;}
int tSM::p06(){ //   CALCS -> CALCS CALC
	return 0;}
int tSM::p07(){ //    CALC -> E
	return 0;}
int tSM::p08(){ //    CALC -> BOOL
	return 0;}
int tSM::p09(){ //    CALC -> STR
	return 0;}
int tSM::p10(){ //    CALC -> DISPSET
	return 0;}
int tSM::p12(){ //       E -> CONST
	return 0;}
int tSM::p13(){ //       E -> ADD
	return 0;}
int tSM::p14(){ //       E -> SUB
	return 0;}
int tSM::p15(){ //       E -> DIV
	return 0;}
int tSM::p16(){ //       E -> MUL
	return 0;}
int tSM::p17(){ //       E -> COND
	return 0;}
int tSM::p18(){ //       E -> IF
	return 0;}
int tSM::p19(){ //       E -> CPROC
	return 0;}
int tSM::p20(){ //   CONST -> $zero
	return 0;}
int tSM::p21(){ //   CONST -> $dec
	return 0;}
int tSM::p22(){ //     ADD -> HADD E )
	return 0;}
int tSM::p23(){ //    HADD -> ( +
	return 0;}
int tSM::p24(){ //    HADD -> HADD E
	return 0;}
int tSM::p25(){ //     SUB -> HSUB E )
	return 0;}
int tSM::p26(){ //    HSUB -> ( -
	return 0;}
int tSM::p27(){ //    HSUB -> HSUB E
	return 0;}
int tSM::p28(){ //     DIV -> HDIV E )
	return 0;}
int tSM::p29(){ //    HDIV -> ( /
	return 0;}
int tSM::p30(){ //    HDIV -> HDIV E
	return 0;}
int tSM::p31(){ //     MUL -> HMUL E )
	return 0;}
int tSM::p32(){ //    HMUL -> ( *
	return 0;}
int tSM::p33(){ //    HMUL -> HMUL E
	return 0;}
int tSM::p34(){ //    COND -> HCOND ELSE )
	return 0;}
int tSM::p35(){ //    COND -> HCOND CLAUS )
	return 0;}
int tSM::p36(){ //   HCOND -> ( cond
	return 0;}
int tSM::p37(){ //   HCOND -> HCOND CLAUS
	return 0;}
int tSM::p38(){ //   CLAUS -> HCLAUS E )
	return 0;}
int tSM::p39(){ //  HCLAUS -> ( BOOL
	return 0;}
int tSM::p40(){ //  HCLAUS -> HCLAUS INTER
	return 0;}
int tSM::p41(){ //    ELSE -> HELSE E )
	return 0;}
int tSM::p42(){ //   HELSE -> ( else
	return 0;}
int tSM::p43(){ //   HELSE -> HELSE INTER
	return 0;}
int tSM::p44(){ //      IF -> ( if BOOL E E )
	return 0;}
int tSM::p48(){ //    BOOL -> $bool
	return 0;}
int tSM::p50(){ //    BOOL -> CPRED
	return 0;}
int tSM::p51(){ //    BOOL -> REL
	return 0;}
int tSM::p52(){ //    BOOL -> OR
	return 0;}
int tSM::p53(){ //    BOOL -> AND
	return 0;}
int tSM::p54(){ //    BOOL -> ( not BOOL )
	return 0;}
int tSM::p60(){ //     REL -> ( < E E )
	return 0;}
int tSM::p61(){ //     REL -> ( = E E )
	return 0;}
int tSM::p62(){ //     REL -> ( > E E )
	return 0;}
int tSM::p63(){ //     REL -> ( >= E E )
	return 0;}
int tSM::p64(){ //     REL -> ( <= E E )
	return 0;}
int tSM::p65(){ //      OR -> HOR BOOL )
	return 0;}
int tSM::p66(){ //     HOR -> ( or
	return 0;}
int tSM::p67(){ //     HOR -> HOR BOOL
	return 0;}
int tSM::p68(){ //     AND -> HAND BOOL )
	return 0;}
int tSM::p69(){ //    HAND -> ( and
	return 0;}
int tSM::p70(){ //    HAND -> HAND BOOL
	return 0;}
int tSM::p71(){ //     STR -> $str
	return 0;}
int tSM::p72(){ //     STR -> SIF
	return 0;}
int tSM::p73(){ //     SIF -> ( if BOOL STR STR )
	return 0;}
int tSM::p75(){ // DISPSET -> ( display E )
	return 0;}
int tSM::p76(){ // DISPSET -> ( display BOOL )
	return 0;}
int tSM::p77(){ // DISPSET -> ( display STR )
	return 0;}
int tSM::p78(){ // DISPSET -> ( newline )
	return 0;}
int tSM::p79(){ // DISPSET -> SET
	return 0;}
int tSM::p80(){ //   INTER -> DISPSET
	return 0;}
int tSM::p81(){ //   INTER -> E
	return 0;}
int tSM::p82(){ //    DEFS -> DEF
	return 0;}
int tSM::p83(){ //    DEFS -> DEFS DEF
	return 0;}
int tSM::p84(){ //     DEF -> PRED
	return 0;}
int tSM::p85(){ //     DEF -> VAR
	return 0;}
int tSM::p86(){ //     DEF -> PROC
	return 0;}
int tSM::p96(){ //   HPROC -> HPROC INTER
	return 0;}
int tSM::p101(){ //    HLET -> HLET INTER
	return 0;}
int tSM::p103(){ //  LETLOC -> LETLOC LETVAR
	return 0;}
//_____________________
int tSM::p105(){return 0;} int tSM::p106(){return 0;} 
int tSM::p107(){return 0;} int tSM::p108(){return 0;} 
int tSM::p109(){return 0;} int tSM::p110(){return 0;} 


