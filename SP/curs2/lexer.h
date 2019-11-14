//                 lexer.h 2019
#ifndef LEXER_H
#define LEXER_H
#include "baselexer.h"
//********************************************
//*          Developed by KAV		         *
//*             (c)  2019                    *
//********************************************
class tLexer : public tBaseLexer{
  public:
	// персональный код разработчика
	std::string Authentication() const {
		return "KAV" + std::string("2019");
	}
	//конструктор
	tLexer() : tBaseLexer(){
		//создать автоматы:

		//  ноль Azero
		addstr(Azero, 0, "+-", 2);
		addstr(Azero, 0, "0", 1);
		addstr(Azero, 2, "0", 1);
		Azero.final(1);
		//________________________________________

		// число
		addstr(Adec, 0, "-+", 1);
		addrange(Adec, 0, '1', '9', 3);
		addstr(Adec, 0, "0", 2);
		addstr(Adec, 0, ".", 4);

		addstr(Adec, 1, "0", 2);
		addstr(Adec, 1, ".", 4);
		addrange(Adec, 1, '1', '9', 3);

		addstr(Adec, 2, ".", 4);
		addstr(Adec, 2, "eE", 6);

		addrange(Adec, 3, '0', '9', 3);
		addstr(Adec, 3, "eE", 6);
		addstr(Adec, 3, ".", 4);

		addrange(Adec, 4, '0', '9', 5);

		addrange(Adec, 5, '0', '9', 5);
		addstr(Adec, 5, "eE", 6);

		addstr(Adec, 6, "-+", 7);

		addrange(Adec, 7, '0', '9', 8);

		addrange(Adec, 8, '0', '9', 8);

		Adec.final(3);
		Adec.final(5);
		Adec.final(8);
		//________________________________________

		// идентификатор
		addstr(Aid, 0, "!", 1);
		addrange(Aid, 0, 'a', 'z', 1);
		addrange(Aid, 0, 'A', 'Z', 1);
		addstr(Aid, 0, "?", 2);

		addstr(Aid, 1, "!", 1);
		addrange(Aid, 1, 'a', 'z', 1);
		addrange(Aid, 1, 'A', 'Z', 1);
		addrange(Aid, 1, '0', '9', 1);
		addstr(Aid, 1, "?", 2);
		addstr(Aid, 1, "-", 3);

		addstr(Aid, 2, "!", 1);
		addrange(Aid, 2, 'a', 'z', 1);
		addrange(Aid, 2, 'A', 'Z', 1);
		addrange(Aid, 2, '0', '9', 1);
		addstr(Aid, 2, "?", 2);
		addstr(Aid, 2, "-", 3);

		addstr(Aid, 3, "!", 1);
		addrange(Aid, 3, 'a', 'z', 1);
		addrange(Aid, 3, 'A', 'Z', 1);
		addrange(Aid, 3, '0', '9', 1);
		addstr(Aid, 3, "-", 3);
		addstr(Aid, 3, "?", 4);

		addstr(Aid, 4, "!", 1);
		addrange(Aid, 4, 'a', 'z', 1);
		addrange(Aid, 4, 'A', 'Z', 1);
		addrange(Aid, 4, '0', '9', 1);
		addstr(Aid, 4, "-", 3);

		Aid.final(1);
		Aid.final(3);
		//________________________________________

		// идентификатор предиката
		addstr(Aidq, 0, "-!", 1);
		addrange(Aidq, 0, 'a', 'z', 1);
		addrange(Aidq, 0, 'A', 'Z', 1);
		addstr(Aidq, 0, "?", 2);

		addstr(Aidq, 1, "!-", 1);
		addrange(Aidq, 1, 'a', 'z', 1);
		addrange(Aidq, 1, 'A', 'Z', 1);
		addrange(Aidq, 1, '0', '9', 1);
		addstr(Aidq, 1, "?", 2);

		addstr(Aidq, 2, "?", 2);
		addstr(Aidq, 2, "-", 1);
		addrange(Aidq, 2, 'a', 'z', 1);
		addrange(Aidq, 2, 'A', 'Z', 1);
		addrange(Aidq, 2, '0', '9', 1);

		Aidq.final(2);
	}
};
#endif