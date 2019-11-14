/* $m19 */
#include "code-gen.h"
using namespace std;
int tCG::p01() {  // S -> PROG
    string header = "/*  " + lex.Authentication() + "   */\n";
    header += "#include \"mlisp.h\"\n";
    header += declarations;
    header += "//________________ \n";
    S1->obj = header + S1->obj;
    return 0;
}
int tCG::p02() {  //  PROG -> CALCS
    S1->obj = "int main(){\n" + S1->obj + "\tstd::cin.get();\n\treturn 0;\n}\n";
    return 0;
}
int tCG::p03() {  //  PROG -> DEFS
    S1->obj +=
        "int main(){\n"
        "\tdisplay(\"No calculations!\");newline();\n"
        "\tstd::cin.get();\n\treturn 0;\n}\n";
    return 0;
}
int tCG::p04() {  //  PROG -> DEFS CALCS
    S1->obj += "int main(){\n" + S2->obj +
	 	"\tstd::cin.get();\n\treturn 0;\n}\n";
    return 0;
}
int tCG::p05() {  // CALCS -> CALC /// хуета какая-то
	S1->obj = S1->obj;
    return 0;
}
int tCG::p06() {  // CALCS -> CALCS CALC
    S1->obj += S2->obj;
    return 0;
}
int tCG::p07() {  //  CALC -> E
    S1->obj = "\tdisplay(" + S1->obj + "); newline();\n";
    return 0;
}
int tCG::p08() {  //     E -> $id
    S1->obj = decor(S1->name);
    return 0;
}
int tCG::p09() {  //     E -> $dec
    //?????????
    S1->obj = decor(S1->name);
    return 0;
}
int tCG::p10() {  //     E -> CPROC /// хуета какая-то
    S1->obj = S1->obj;
    return 0;
}
int tCG::p11() {  //     E -> MUL /// хуета какая-то
    S1->obj = S1->obj;
    return 0;
}
int tCG::p12() {  // CPROC -> HCPROC )
    S1->obj += ")";
    return 0;
}
int tCG::p13() {  //HCPROC -> ( $id
    S1->obj = decor(S2->name) + "(";
    return 0;
}
int tCG::p14() {  //HCPROC -> HCPROC E
	if(S1->count > 0)
		S1->obj += ", ";
    S1->obj += S2->obj;
	S1->count++;
    return 0;
}
int tCG::p15() {         //   MUL -> HMUL E )
    if (S1->count == 0)  //���� �������
        S1->obj = S2->obj;
    else  //����� ������ ��������
        S1->obj += S2->obj;
    S1->count = 0;
    return 0;
}
int tCG::p16() {  //  HMUL -> ( * /// хуета какая-то
    S1->obj = "";
    return 0;
}
int tCG::p17() {  //  HMUL -> HMUL E
    S1->obj += S2->obj + " * ";
    ++S1->count;
    return 0;
}
int tCG::p18() {  //   SET -> ( set! $id E )
    S1->obj = "\t" + decor(S3->name) + " = " + S4->obj + ";"; 
    return 0;
}
int tCG::p19() {  //   DEF -> PROC /// хуета какая-то
    S1->obj = S1->obj;
    return 0;
}
int tCG::p20() {  //  DEFS -> DEF
    S1->obj = S1->obj;
    return 0;
}
int tCG::p21() {  //  DEFS -> DEFS DEF
                  //????
    S1->obj += S2->obj;
    return 0;
}
int tCG::p22() {  //  PROC -> HPROC E )
                  //????
	S1->obj += "\treturn " + S2->obj + ";\n}\n";
    return 0;
}
int tCG::p23() {  // HPROC -> PCPAR )
    S1->obj += ")";
    declarations += S1->obj + ";\n";
	S1->obj += "{\n";
    //????
    return 0;
}
int tCG::p24() {  // HPROC -> HPROC SET
                  //????
	S1->obj += S2->obj + "\n";
    return 0;
}
int tCG::p25() {  // PCPAR -> ( define ( $id
                  //????
	S1->obj = "double " + decor(S4->name) + "(";
    return 0;
}
int tCG::p26() {  // PCPAR -> PCPAR $id
    if(S1->count > 0)
		S1->obj += ", ";
	S1->obj += "double " + decor(S2->name);
	S1->count++;
    return 0;
}
//_____________________
int tCG::p27() { return 0; }
int tCG::p28() { return 0; }
int tCG::p29() { return 0; }
int tCG::p30() { return 0; }
int tCG::p31() { return 0; }
int tCG::p32() { return 0; }
int tCG::p33() { return 0; }
int tCG::p34() { return 0; }
int tCG::p35() { return 0; }
int tCG::p36() { return 0; }
int tCG::p37() { return 0; }
int tCG::p38() { return 0; }
int tCG::p39() { return 0; }
int tCG::p40() { return 0; }
int tCG::p41() { return 0; }
int tCG::p42() { return 0; }
int tCG::p43() { return 0; }
int tCG::p44() { return 0; }
int tCG::p45() { return 0; }
int tCG::p46() { return 0; }
int tCG::p47() { return 0; }
int tCG::p48() { return 0; }
int tCG::p49() { return 0; }
int tCG::p50() { return 0; }
int tCG::p51() { return 0; }
int tCG::p52() { return 0; }
int tCG::p53() { return 0; }
int tCG::p54() { return 0; }
int tCG::p55() { return 0; }
int tCG::p56() { return 0; }
int tCG::p57() { return 0; }
int tCG::p58() { return 0; }
int tCG::p59() { return 0; }
int tCG::p60() { return 0; }
int tCG::p61() { return 0; }
int tCG::p62() { return 0; }
int tCG::p63() { return 0; }
int tCG::p64() { return 0; }
int tCG::p65() { return 0; }
int tCG::p66() { return 0; }
int tCG::p67() { return 0; }
int tCG::p68() { return 0; }
int tCG::p69() { return 0; }
int tCG::p70() { return 0; }
int tCG::p71() { return 0; }
int tCG::p72() { return 0; }
int tCG::p73() { return 0; }
int tCG::p74() { return 0; }
int tCG::p75() { return 0; }
int tCG::p76() { return 0; }
int tCG::p77() { return 0; }
int tCG::p78() { return 0; }
int tCG::p79() { return 0; }
int tCG::p80() { return 0; }
int tCG::p81() { return 0; }
int tCG::p82() { return 0; }
int tCG::p83() { return 0; }
int tCG::p84() { return 0; }
int tCG::p85() { return 0; }
int tCG::p86() { return 0; }
int tCG::p87() { return 0; }
int tCG::p88() { return 0; }
int tCG::p89() { return 0; }
int tCG::p90() { return 0; }
int tCG::p91() { return 0; }
int tCG::p92() { return 0; }
int tCG::p93() { return 0; }
int tCG::p94() { return 0; }
int tCG::p95() { return 0; }
int tCG::p96() { return 0; }
int tCG::p97() { return 0; }
int tCG::p98() { return 0; }
int tCG::p99() { return 0; }
int tCG::p100() { return 0; }
int tCG::p101() { return 0; }
int tCG::p102() { return 0; }
int tCG::p103() { return 0; }
int tCG::p104() { return 0; }
int tCG::p105() { return 0; }
int tCG::p106() { return 0; }
int tCG::p107() { return 0; }
int tCG::p108() { return 0; }
int tCG::p109() { return 0; }
int tCG::p110() { return 0; }
