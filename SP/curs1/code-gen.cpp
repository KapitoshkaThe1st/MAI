/* $b09 */
#include "code-gen.h"

#define TAB_SIZE 4 // количество пробелов в одном знаке табуляции

const std::string oneTab(TAB_SIZE, ' '); // чтобы не конструировать строку несколько раз
std::string globalVars; // накапливает определения глобальных переменных
int tabCount = 0;       // означает количество знаков табуляции, нужное для формирования кода с текущим уровнем вложенности 

using namespace std;
int tCG::p01() {  // S -> PROG
    string header = "/*  " + lex.Authentication() + "   */\n";
    header += "#include \"mlisp.h\"\n\n";
    header += globalVars + "\n";

    // clearing before the next translation
    globalVars.clear();
    tabCount = 0;

    header += declarations;
    header += "//________________ \n";
    S1->obj = header + S1->obj;
    return 0;
}
int tCG::p02() {  //   PROG -> CALCS
    S1->obj = "int main(){\n" + S1->obj + oneTab + "std::cin.get();\n" + oneTab + "return 0;\n}\n";
    return 0;
}
int tCG::p03() {  //   PROG -> DEFS
    S1->obj += "int main(){\n";
    S1->obj.append(TAB_SIZE, ' ');
    S1->obj += "display(\"No calculations!\");newline();\n";
    S1->obj.append(TAB_SIZE, ' ');
    S1->obj += "std::cin.get();\n";
    S1->obj.append(TAB_SIZE, ' ');
    S1->obj += "return 0;\n}\n";
    return 0;
}
int tCG::p04() {  //   PROG -> DEFS CALCS
    S1->obj += "int main(){\n";
    S1->obj += S2->obj;
    S1->obj.append(TAB_SIZE, ' ');
    S1->obj += "std::cin.get();\n";
    S1->obj.append(TAB_SIZE, ' ');
    S1->obj += "return 0;\n}\n";
    return 0;
}
int tCG::p05() {  //  CALCS -> CALC
    return 0;
}
int tCG::p06() {  //  CALCS -> CALCS CALC
    S1->obj += S2->obj;
    return 0;
}
int tCG::p07() {  //   CALC -> E
    S1->obj = oneTab + "display(" + S1->obj + "); newline();\n";
    return 0;
}
int tCG::p08() {  //   CALC -> BOOL
    S1->obj = oneTab + "display(" + S1->obj + "); newline();\n";
    return 0;
}
int tCG::p09() {  //   CALC -> STR
    S1->obj = oneTab + "display(" + S1->obj + "); newline();\n";
    return 0;
}
int tCG::p10() {  //   CALC -> DISPSET
    S1->obj = oneTab + S1->obj + ";\n"; 
    return 0;
}
int tCG::p11() {  //      E -> $id
    S1->obj = decor(S1->name);
    return 0;
}
int tCG::p12() {  //      E -> $zero
    S1->obj = decor(S1->name);
    return 0;
}
int tCG::p13() {  //      E -> ADD
    return 0;
}
int tCG::p14() {  //      E -> SUB
    return 0;
}
int tCG::p15() {  //      E -> DIV
    return 0;
}
int tCG::p16() {  //      E -> MUL
    return 0;
}
int tCG::p17() {  //      E -> COND
    return 0;
}
int tCG::p18() {  //      E -> CPROC
    return 0;
}
int tCG::p19() {  //    ADD -> HADD E )
    S1->obj += (S1->count > 0 ? " + " : "+");
    S1->obj += S2->obj;
    S1->obj += ")";
    return 0;
}
int tCG::p20() {  //   HADD -> ( +
    S1->obj = "("; 
    return 0;
}
int tCG::p21() {  //   HADD -> HADD E
    if(S1->count > 0)
        S1->obj += " + ";
    S1->obj += S2->obj; 
    S1->count++;
    return 0;
}
int tCG::p22() {  //    SUB -> HSUB E )
    S1->obj += (S1->count > 0 ? " - " : "-");
    S1->obj += S2->obj;
    S1->obj += ")";
    return 0;
}
int tCG::p23() {  //   HSUB -> ( -
    S1->obj = "("; 
    return 0;
}
int tCG::p24() {  //   HSUB -> HSUB E
    if (S1->count > 0)
        S1->obj += " - ";
    S1->obj += S2->obj; 
    S1->count++;
    return 0;
}
int tCG::p25() {  //    DIV -> HDIV E )
    if(S1->count > 0){
        S1->obj += " / ";
    }
    S1->obj += S2->obj;
    S1->obj += ")";
    return 0;
}
int tCG::p26() {  //   HDIV -> ( /
    S1->obj = "("; 
    return 0;
}
int tCG::p27() {  //   HDIV -> HDIV E
    if (S1->count > 0)
        S1->obj += " / ";
    if(S1->count == 0){
        S1->obj += "double(";
        S1->obj += S2->obj;
        S1->obj += ")";
    }
    else{
        S1->obj += S2->obj; 
    }
    S1->count++;
    return 0;
}
int tCG::p28() {  //    MUL -> HMUL E )
    if (S1->count == 0)
        S1->obj = S2->obj;
    else
        S1->obj += S2->obj;
    S1->obj += ")";
    S1->count = 0;
    return 0;
}
int tCG::p29() {  //   HMUL -> ( *
    S1->obj = "(";
    return 0;
}
int tCG::p30() {  //   HMUL -> HMUL E
    S1->obj += S2->obj;
    S1->obj += " * ";
    ++S1->count;
    return 0;
}
int tCG::p31() {  //   COND -> HCOND CLAUS )
    S1->obj += S2->obj;
    S1->obj += "\n";
    S1->obj.append(tabCount, ' ');
    S1->obj += ": _infinity";
    S1->obj.append(S1->count + 1, ')');

    tabCount -= (S1->count + 1) * TAB_SIZE;
    
    S1->obj += ")";
    return 0;
}
int tCG::p32() {  //  HCOND -> ( cond
    S1->obj = "(";
    tabCount += TAB_SIZE;
    return 0;
}
int tCG::p33() {  //  HCOND -> HCOND CLAUS
    S1->obj += S2->obj;
    S1->obj += "\n";
    S1->obj.append(tabCount, ' ');
    S1->obj += ": ";
    S1->count++;
    tabCount += TAB_SIZE;
    return 0;
}
int tCG::p34() {  //  CLAUS -> HCLAUS E )
    if (S1->count > 0) {
        S1->obj.append(tabCount, ' ');
    }
    S1->obj += S2->obj;
    tabCount -= TAB_SIZE;
    return 0;
}
int tCG::p35() {  // HCLAUS -> ( BOOL
    S1->obj = "(" + S2->obj + " ? ";
    tabCount += TAB_SIZE;
    return 0;
}
int tCG::p36() {  // HCLAUS -> HCLAUS INTER
    if(S1->count > 0){
        S1->obj.append(tabCount, ' ');
    }
    S1->obj += S2->obj + ",\n";
    S1->count++;

    return 0;
}
int tCG::p37() {  //   ELSE -> HELSE E )
    return 0;
}
int tCG::p38() {  //  HELSE -> ( else
    return 0;
}
int tCG::p39() {  //  HELSE -> HELSE INTER
    return 0;
}
int tCG::p40() {  //  CPROC -> HCPROC )
    S1->obj += ")";
    return 0;
}
int tCG::p41() {  // HCPROC -> ( $id
    S1->obj = decor(S2->name) + "(";
    return 0;
}
int tCG::p42() {  // HCPROC -> HCPROC E
    if (S1->count > 0)
        S1->obj += ", ";
    S1->obj += S2->obj;
    S1->count++;
    return 0;
}
int tCG::p43() {  //   BOOL -> $bool
    S1->obj = (S1->name == "#t" ? "true" : "false");
    return 0;
}
int tCG::p44() {  //   BOOL -> $idq
    S1->obj = decor(S1->name);
    return 0;
}
int tCG::p45() {  //   BOOL -> CPRED
    return 0;
}
int tCG::p46() {  //   BOOL -> REL
    return 0;
}
int tCG::p47() {  //  CPRED -> HCPRED )
    S1->obj += ")";
    return 0;
}
int tCG::p48() {  // HCPRED -> ( $idq
    S1->obj = decor(S2->name) + "(";
    return 0;
}
int tCG::p49() {  // HCPRED -> HCPRED ARG
    if(S1->count > 0)
        S1->obj += ", ";
    S1->obj += S2->obj;
    S1->count++;
    return 0;
}
int tCG::p50() {  //    ARG -> E
    return 0;
}
int tCG::p51() {  //    ARG -> BOOL
    return 0; 
}
int tCG::p52() {  //    REL -> ( < E E )
    S1->obj = "(" + S3->obj + " < " + S4->obj + ")";
    return 0;
}
int tCG::p53() {  //    REL -> ( = E E )
    S1->obj = "(" + S3->obj + " == " + S4->obj + ")";
    return 0;
}
int tCG::p54() {  //    STR -> $str
    S1->obj = S1->name; 
    return 0;
}
int tCG::p55() {  //    STR -> SIF
    return 0;
}
int tCG::p56() {  //    SIF -> ( if BOOL STR STR )
    S1->obj = "(" + S3->obj + " ? " + S4->obj + " : " + S5->obj + ")";
    return 0;
}
int tCG::p57() {  //    SET -> ( set! $id E )
    S1->obj = decor(S3->name) + " = " + S4->obj;
    return 0;
}
int tCG::p58() {  //	DISPSET -> ( display E )
    S1->obj = "display(" + S3->obj + ")"; 
    return 0;
}
int tCG::p59() {  //	DISPSET -> ( display BOOL )
    S1->obj = "display(" + S3->obj + ")"; 
    return 0;
}
int tCG::p60() {  //	DISPSET -> ( display STR )
    S1->obj = "display(" + S3->obj + ")"; 
    return 0;
}
int tCG::p61() {  //	DISPSET -> ( newline )
    S1->obj = "newline()"; 
    return 0;
}
int tCG::p62() {  //	DISPSET -> SET
    return 0;
}
int tCG::p63() {  //  INTER -> DISPSET
    return 0;
}
int tCG::p64() {  //  INTER -> E
    return 0;
}
int tCG::p65() {  //   DEFS -> DEF
    return 0;
}
int tCG::p66() {  //   DEFS -> DEFS DEF
    S1->obj += S2->obj;
    return 0;
}
int tCG::p67() {  //    DEF -> PRED
    return 0;
}
int tCG::p68() {  //    DEF -> VAR
    return 0;
}
int tCG::p69() {  //    DEF -> PROC
    return 0;
}
int tCG::p70() {  //   PRED -> HPRED BOOL )
    S1->obj += S2->obj;
    S1->obj += ";\n}\n\n";
    tabCount -= TAB_SIZE;
    return 0;
}
int tCG::p71() {  //  HPRED -> PDPAR )
    S1->obj += ")";
    declarations += S1->obj + ";\n";  //!!!
    S1->obj += "{\n";
    S1->obj.append(TAB_SIZE, ' ');
    S1->obj += "return ";
    S1->count = 0;
    return 0;
}
int tCG::p72() {  //  PDPAR -> ( define ( $idq
    S1->obj = "bool " + decor(S4->name) + "(";
    S1->count = 0;
    tabCount += TAB_SIZE;
    return 0;
}
int tCG::p73() {  //  PDPAR -> PDPAR $idq
    if (S1->count) 
        S1->obj += ", ";
    S1->obj += "bool ";
    S1->obj += decor(S2->name);
    ++(S1->count);
    return 0;
}
int tCG::p74() {  //  PDPAR -> PDPAR $id
    if (S1->count)
        S1->obj += ", ";
    S1->obj += "double ";
    S1->obj += decor(S2->name);
    ++(S1->count);
    return 0;
}
int tCG::p75() {  //  CONST -> $zero
    S1->obj = "0"; 
    return 0;
}
int tCG::p76() {  //  CONST -> $dec
    S1->obj = decor(S1->name); 
    return 0;
}
int tCG::p77() {  //    VAR -> ( define $id CONST )
    globalVars += "double " + decor(S3->name) + " = " + S4->obj + ";\n";
    S1->obj = "";
    return 0;
}
int tCG::p78() {  //   PROC -> HPROC LET )
    S1->obj += S2->obj;
    S1->obj += "}\n\n";
    tabCount -= TAB_SIZE;
    return 0;
}
int tCG::p79() {  //   PROC -> HPROC E )
    S1->obj.append(TAB_SIZE, ' ');
    S1->obj += "return ";
    S1->obj += S2->obj;
    S1->obj += ";\n}\n\n";
    tabCount -= TAB_SIZE;
    return 0;
}
int tCG::p80() {  //  HPROC -> PCPAR )
    S1->obj += ")";
    declarations += S1->obj + ";\n";
    S1->obj += "{\n";

    tabCount += TAB_SIZE;

    return 0;
}
int tCG::p81() {  //  HPROC -> HPROC INTER
    S1->obj.append(tabCount, ' ');
    S1->obj += S2->obj;
    S1->obj += ";\n";
    return 0;
}
int tCG::p82() {  //  PCPAR -> ( define ( $id
    S1->obj = "double " + decor(S4->name) + "(";
    return 0;
}
int tCG::p83() {  //  PCPAR -> PCPAR $id
    if (S1->count > 0)
        S1->obj += ", ";
    S1->obj += "double ";
    S1->obj += decor(S2->name);
    S1->count++;
    return 0;
}
int tCG::p84() {  //    LET -> HLET E )
    S1->obj.append(tabCount, ' ');
    S1->obj += "return ";
    S1->obj += S2->obj;
    S1->obj += ";\n";
    S1->obj.append(TAB_SIZE, ' ');
    S1->obj += "}\n";
    tabCount -= TAB_SIZE;
    return 0;
}
int tCG::p85() {  //   HLET -> LETLOC )
    S1->obj = oneTab + "{\n" + S1->obj + ";\n";
    return 0;
}
int tCG::p86() {  //   HLET -> HLET INTER
    S1->obj.append(tabCount, ' ');
    S1->obj += S2->obj;
    S1->obj += ";\n";
    return 0;
}
int tCG::p87() {  // LETLOC -> ( let (
    tabCount += TAB_SIZE;
    S1->obj = std::string(tabCount, ' ') + "double " + S1->obj;
    return 0;
}
int tCG::p88() {  // LETLOC -> LETLOC LETVAR
    if(S1->count > 0){
        S1->obj += ",\n";
        S1->obj.append(tabCount, ' ');
    }
    S1->obj += S2->obj;
    S1->count++;
    return 0;
}
int tCG::p89() {  // LETVAR -> ( $id E )
    S1->obj = decor(S2->name) + "(" + S3->obj + ")";
    return 0;
}
//_____________________
int tCG::p90(){return 0;} int tCG::p91(){return 0;} 
int tCG::p92(){return 0;} int tCG::p93(){return 0;} 
int tCG::p94(){return 0;} int tCG::p95(){return 0;} 
int tCG::p96(){return 0;} int tCG::p97(){return 0;} 
int tCG::p98(){return 0;} int tCG::p99(){return 0;} 
int tCG::p100(){return 0;} int tCG::p101(){return 0;} 
int tCG::p102(){return 0;} int tCG::p103(){return 0;} 
int tCG::p104(){return 0;} int tCG::p105(){return 0;} 
int tCG::p106(){return 0;} int tCG::p107(){return 0;} 
int tCG::p108(){return 0;} int tCG::p109(){return 0;} 
int tCG::p110(){return 0;} 
