/* $mlisp19 */
#include "semantics.h"
#include "semempty.cpp"

#include <iostream>
#include <sstream>

using namespace std;

const string numEx = "[numeric expression]";
const string boolEx = "[boolean expression]";

void tSM::setDef(tgName &r) {
    if (r.empty()) {
        r = tgName(VAR | USED);
    } else {
        r.set(USED);
    }
}

void tSM::formBuf(ostringstream &buf, const std::vector<std::string> &vec) {
    int count = vec.size();
    for (int i = 0; i < count; i++) {
        buf << vec[i];
        if (i != count - 1)
            buf << ", ";
    }
    buf << endl;
}

// вызывается при начале обработки нового текста программы
void tSM::init() {
    globals.clear();
    locals.clear();
    params.clear();
    scope = 0;
    //константы:
    globals["e"] =
        tgName(VAR | DEFINED | BUILT);
    globals["pi"] =
        tgName(VAR | DEFINED | BUILT);

    // элементарные процедуры:
    globals["remainder"] =
        tgName(PROC | DEFINED | BUILT, 2);
    globals["quotient"] =
        tgName(PROC | DEFINED | BUILT, 2);
    globals["abs"] =
        tgName(PROC | DEFINED | BUILT, 1);
    globals["expt"] =
        tgName(PROC | DEFINED | BUILT, 2);
    globals["log"] =
        tgName(PROC | DEFINED | BUILT, 1);
    globals["atan"] =
        tgName(PROC | DEFINED | BUILT, 1);
    globals["sqrt"] =
        tgName(PROC | DEFINED | BUILT, 1);
    return;
}
int tSM::p01() {  //       S -> PROG
    bool error = false;

    vector<string> undefProc;
    vector<string> undefVar;

    vector<string> unusedProc;
    vector<string> unusedVar;

    for (tGlobal::iterator it = globals.begin();
         it != globals.end();
         ++it) {
        const string &name = it->first;
        tgName &ref = it->second;

        if (ref.test(PROC)) {
            if (!ref.test(DEFINED)) {
                undefProc.push_back(name);
                error = true;
            }
            if (!ref.test(USED) && !ref.test(BUILT)) {
                unusedProc.push_back(name);
            }
        } 
        else if (ref.test(VAR)) {
            if (!ref.test(DEFINED)) {
                undefVar.push_back(name);
                error = true;
            }
            if (!ref.test(USED) && !ref.test(BUILT)) {
                unusedVar.push_back(name);
            }
        }
    }

    ostringstream buf;

    /*auto formBuf = [&buf](const vector<string> &vec) {
        int count = vec.size();
        for (int i = 0; i < count; i++) {
            buf << vec[i];
            if (i != count - 1)
                buf << ", ";
        }
        buf << endl;
    };*/

    if (!undefProc.empty()) {
        buf << "[!]Undefined procedures: ";
        formBuf(buf, undefProc);
    }
    if (!undefVar.empty()) {
        buf << "[!]Undefined variables: ";
        formBuf(buf, undefVar);
    }
    if (!unusedProc.empty()) {
        buf << "[?]Unused procedures: ";
        formBuf(buf, unusedProc);
    }
    if (!unusedVar.empty()) {
        buf << "[?]Unused variables: ";
        formBuf(buf, unusedVar);
    }

    ferror_message = buf.str();
    if (error){
        return 1;
    }

    return 0;
}

int tSM::p11() {  //       E -> $id
    const string &name = S1->name;
    tgName &ref = globals[name];

    ostringstream buf;
    buf << "[!]Numeric expression evaluation: '" << S1->name << "' ";

    bool isProc = ref.test(PROC);

    /*auto setDef = [](tgName &r) {
        if (r.empty()) {
            r = tgName(VAR | USED);
        }
        else {
            r.set(USED);
        }
    };*/

    if (scope == 0) { 
        if(isProc){ // p11-1
            buf <<  "is procedure not a variable !";
            ferror_message = buf.str();
            return 1;
        }
        setDef(ref); // p11-2
    }

    if(scope == 1){
        if(!params.count(name)){ 
            if(isProc){ // p11-4
                buf << "is procedure not a variable !";
                ferror_message = buf.str();
                return 1;
            }
            setDef(ref); // p11-5
        } // p11-3
    }

    if(scope == 2){
        if(!locals.count(name)){
            if (!params.count(name)) {
                if (isProc) { // p11-8
                    buf << "is procedure not a variable !";
                    ferror_message = buf.str(); 
                    return 1;
                }
                setDef(ref); // p11-9
            } // p11-7
        } // p11-6
    }

    return 0;
}
int tSM::p45() {  //   CPROC -> HCPROC )
    string name = S1->name;
    int count = S1->count;
    if (scope > 1) {               // внури тела let
        if (locals.count(name)) {  // локальное имя
                                   //p45-1.ss
            ferror_message =
                "[!]Procedure application:"
                " local variable '" +
                name +
                "' shadows the procedure!";
            return 1; // 1 если ошибка
        }                          // if locals ...
    }                              // if scope ...
    if (scope > 0) {               // внутри процедуры
        if (params.count(name)) {  // имя параметра
                                   //p45-2.ss
            ferror_message =
                "[!]Procedure application:"
                " parameter '" +
                name +
                "' shadows the procedure!";
            return 1;
        }  // if params...
    }      // if scope...

    // do {} while(false) для удобства, чтобы можно было пользоваться break'ами.
    do {
        //  найти имя в глобальной таблице
        tgName& ref = globals[name];
        if (ref.empty()) {  //неизвестное имя
                            //        создать новую учетную запись
                            // cout << name << " USED WITH " << count << endl;
            ref = tgName(PROC | USED, count);
            break;
        }

        // имя найдено
        if (!ref.test(PROC)) {  //не процедура
                                //p45-3.ss
            ferror_message =
                "[!]Procedure application:"
                "  '" +
                name +
                "' is not a procedure!";
            return 1;
        }

        if (ref.arity != count) {  //число аргументов
                                   //                не равно числу параметров
            std::ostringstream buf;
            buf << "[!]Procedure application: '" << name << "' "
                //p45-4.ss
                << (ref.test(DEFINED) ? "expects "  // процедура
                                                    //                                      уже определена
                                                    //p45-5.ss

                                      // процедура еще не определена, но уже вызывалась ранее
                                      : "has been called already\n\t with ")
                << ref.arity << " argument"
                << (ref.arity != 1 ? "s" : "")
                << ", given: " << count << " !";
            ferror_message = buf.str();
            return 1;
        }
        // ошибок нет
        ref.set(USED);  //имя использовано
    } while (false);
    return 0;
}
int tSM::p46() {  //  HCPROC -> ( $id
    S1->name = S2->name;
    S1->count = 0;
    return 0;
}
int tSM::p47() {  //  HCPROC -> HCPROC E
    ++S1->count;
    return 0;
}
int tSM::p49() {  //    BOOL -> $idq
    const string &name = S1->name;

    if (!params.count(name)) { // p49-1
        ferror_message =
            "[!]Boolean expression evaluation: '" + S1->name + "' is inaccessible in this scope!";
        return 1;
    }
    // p49-2
    return 0;
}
int tSM::p55() {  //   CPRED -> HCPRED ) // non-checked
    string name = S1->name;
    int count = S1->count;
    int types = S1->types;

    if (scope > 0) {
        if (params.count(name)) { // p55-1
            ferror_message =
                "[!]Predicate application:"
                " parameter '" +
                name +
                "' shadows the predicate!";
            return 1;
        }
    }

    do {
        //  найти имя в глобальной таблице
        tgName& ref = globals[name];
        if (ref.empty()) { // p55-2
            ref = tgName(PROC | USED, count, types);
            break;
        }

        if (ref.arity != count) { // p55-3
            std::ostringstream buf;
            buf << "[!]Predicate application: '" << name << "' "
                //p45-4.ss
                << (ref.test(DEFINED) ? "expects "
                                      : "has been called already\n\t with ")
                << ref.arity << " argument"
                << (ref.arity != 1 ? "s" : "")
                << ", given: " << count << " !";
            ferror_message = buf.str();
            return 1;
        }

        if(ref.types != types){ // p55-4
            int diffInd;
            for(int i = 0; i < sizeof(types); i++){
                if ((ref.types & (1 << i)) != (types & (1 << i))){
                    diffInd = i;
                    break;
                }
            }

            int type = (ref.types & (1 << diffInd));

            std::ostringstream buf;
            buf << "[!]Predicate application: '" << name << "' "
                << (ref.test(DEFINED) ? "expects "
                                      : "has been called already\n\t with ")
                << (type ? boolEx : numEx) << " on position " << diffInd
                << ", given: " << (!type ? boolEx : numEx) << " !";
            ferror_message = buf.str();
            return 1;
        }

        ref.set(USED);
    } while (false);

    // p55-5
    return 0;
}
int tSM::p56() {  //  HCPRED -> ( $idq
    S1->name = S2->name;
    S1->types = 0; // non-checked
    S1->count = 0;
    return 0;
}
int tSM::p57() {  //  HCPRED -> HCPRED ARG // non-checked

    S1->types |= (S2->types << S1->count);
    S1->count++;
    return 0;
}
int tSM::p58() {  //     ARG -> E
    S1->types = 0; // non-checked
    return 0;
}
int tSM::p59() {  //     ARG -> BOOL
    S1->types = 1; // non-checked
    return 0;
}
int tSM::p74() {  //     SET -> ( set! $id E )
    const string &name = S3->name;
//    tgName &ref = globals[name];

	tGlobal::iterator it = globals.find(name);

    bool isProc = false;
    bool isBuilt = false;
    if(it != globals.end()){
    	isProc = it->second.test(PROC);
    	isBuilt = it->second.test(BUILT);
    }

    ostringstream buf;
    buf << "[!]Assignment operation: '"
    << name << "' is ";

    if (isBuilt) {
        buf << "built-in ";
        if (isProc) {
            buf << "procedure";
        }
        else {
            buf << "constant";
        }
    }
    else if(isProc) {
        buf << "procedure";
    }

    /*auto setDef = [](tgName &r) {
        if (r.empty()) {
            r = tgName(VAR | USED);
        } else {
            r.set(USED);
        }
    };*/

    if (scope == 0) { 
        if (isProc || isBuilt) {  // p74-1
            ferror_message = buf.str();
            return 1;
        }
        setDef(globals[name]); // p74-2
    }

    if (scope == 1) {  
        if (!params.count(name)) {
            if (isProc || isBuilt) {  // p74-3
                ferror_message = buf.str();
                return 1;
            }
            setDef(globals[name]); // p74-4
        }
        // p74-5
    }

    if (scope == 2) {
        if (!locals.count(name)) {
            if (!params.count(name)) {
                // cout << "here" << endl;
                if (isProc || isBuilt) { // p74-6
                    ferror_message = buf.str();
                    return 1;
                }
                setDef(globals[name]); // p74-7
            }
            // p74-8
        }
        // p74-9
    }

    return 0;
}

int tSM::p87() {  //    PRED -> HPRED BOOL )
    scope = 0;
    params.clear();
    return 0;
}
int tSM::p88() {  //   HPRED -> PDPAR )
    scope = 1;
    const string &name = S1->name;
    tgName &ref = globals[name];
    int arity = S1->count;
    int types = S1->types;

    if (ref.empty()) { // p88-1
        ref = tgName(PROC | DEFINED, arity, types);
        return 0;
    }

    std::ostringstream buf;

    buf << "[!]Predicate definition: predicate '"
        << name << "' ";

    if (ref.test(DEFINED)) { // p88-2
        buf << "has been aldeady defined";
        ferror_message = buf.str();
        return 1;
    }

    if (ref.test(USED)) {
        if (ref.arity != S1->count) { // p88-3
            buf << "has been already called\n\t with "
                << ref.arity << " arguments"
                << ", given: " << S1->count << " !";
            ferror_message = buf.str();
            return 1;
        }

        if (S1->types != ref.types) { // p88-4
            int diffInd;
            for (int i = 0; i < sizeof(S1->types); i++) {
                if ((ref.types & (1 << i)) != (S1->types & (1 << i))) {
                    diffInd = i;
                    break;
                }
            }

            int type = ref.types & (1 << diffInd);

            buf << "has been already called\n\t with "
                << (type ? boolEx : numEx)
                << " on position " << diffInd
                << ", given: " << (!type ? boolEx : numEx) << " !";

            ferror_message = buf.str();
            return 1;
        }
        // p88-5
    }

    ref.set(DEFINED);
    return 0;
}
int tSM::p89() {  //   PDPAR -> ( define ( $idq
    S1->types = 0;
    S1->name = S4->name; // non-checked
    return 0;
}
int tSM::p90() {  //   PDPAR -> PDPAR $idq
    if (params.count(S2->name)) { //p90-1
        ferror_message =
            "[!]Procedure definition: in '" + S1->name +
            "' duplicate parameter identifier '" + S2->name + "'!";
        return 1;
    }
    params.insert(S2->name);
    S1->types |= (1 << S1->count);
    S1->count++; //p90-1
    return 0;
}
int tSM::p91() {  //   PDPAR -> PDPAR $id
    if (params.count(S2->name)) { //p91-1
        ferror_message =
            "[!]Procedure definition: in '" + S1->name +
            "' duplicate parameter identifier '" + S2->name + "'!";
        return 1;
    }
    params.insert(S2->name);
    S1->count++; //p91-2
    return 0;
}
int tSM::p92() {  //     VAR -> ( define $id CONST )
    const string &name = S3->name;
    tgName &ref = globals[name];

    if (ref.empty()) { // p92-1
        ref = tgName(VAR | DEFINED);
        return 0;
    }

    std::ostringstream buf;

    buf << "[!]Global variable definition: ";

    if (ref.test(PROC)) {
        buf << "procedure";
    } else {
        buf << "variable";
    }

    buf << " '" << name << "' ";

    if (ref.test(DEFINED)) { // p92-2
        buf << "has been aldeady defined";
        ferror_message = buf.str();
        return 1;
    } 
    else if (ref.test(USED)) {
        if (ref.test(PROC)) { // p92-3
            buf << "has been already used";
            ferror_message = buf.str();
            return 1;
        }
    }
    ref.set(DEFINED); // p92-4
    return 0;
}
int tSM::p93() {  //    PROC -> HPROC LET )
    params.clear();
    scope = 0;
    return 0;
}
int tSM::p94() {  //    PROC -> HPROC E )
    params.clear();
    scope = 0;
    return 0;
}
int tSM::p95() {  //   HPROC -> PCPAR )

    scope = 1;
    const string &name = S1->name;
    tgName &ref = globals[name];
    int arity = S1->count;

    if (ref.empty()) { // p95-1
        ref = tgName(PROC | DEFINED, arity);
        return 0;
    }

    std::ostringstream buf;

    buf << "[!]Procedure definition: ";

    if(ref.test(PROC)){
        buf << "procedure";
    }
    else{
        buf << "variable";
    }

    buf << " '" << name << "' ";

    if(ref.test(DEFINED)){ // p95-2
        buf << "has been aldeady defined";
        ferror_message = buf.str();
        return 1;
    }
    else if(ref.test(USED)){
        if (ref.test(PROC)) {
            if (ref.arity != arity) { // p95-3
                buf << "has been already called\n\t with "
                    << ref.arity << " arguments"
                    << ", given: " << S1->count << " !";
                ferror_message = buf.str();
                return 1;
            }
        }
        else{ // p95-4
            buf << "has been already used";
            ferror_message = buf.str();
            return 1;
        }
    }

    ref.set(DEFINED); // p95-5
    return 0;
}

int tSM::p97() {  //   PCPAR -> ( define ( $id
    S1->name = S4->name;
    S1->count = 0;
    return 0;
}

// проверка на повторяющиеся параметры
int tSM::p98() {  //   PCPAR -> PCPAR $id
    if (params.count(S2->name)) { //p98-1
        ferror_message =
            "[!]Procedure definition: in '" + S1->name +
            "' duplicate parameter identifier '" + S2->name + "'!";
        return 1;
    }
    params.insert(S2->name);
    ++S1->count;  //p98a
    return 0;
}
int tSM::p99() {  //     LET -> HLET E )
    locals.clear();
    return 0;
}
int tSM::p100() {  //    HLET -> LETLOC )
    scope = 2;
    return 0;
}
int tSM::p102() {  //  LETLOC -> ( let (
    locals.clear();
    return 0;
}
int tSM::p104() {  //  LETVAR -> ( $id E )
    if (locals.count(S2->name)) { //p104-1
        ferror_message =
            "[!]Local variable definition:"
            " duplicate local vatiable identifier '"
            + S2->name + "'!";
        return 1;
    }
    locals.insert(S2->name);  //p104-2
    return 0;
}
//_____________________
