#ifndef FSM_H
#define FSM_H

#include <vector>
#include <map>
#include <set>

#include <iostream>

using namespace std;

class tFSM{
  public:
	// типы
	typedef char tSymbol;
	typedef unsigned char tState;
	typedef std::set<tState> tStateSet;
	// конструктор
	tFSM(){}; //создает "пустой" автомат
			  // функции-члены
			  // добавляет одну команду(from,c)->to
	void add(tState from, tSymbol c, tState to);
	void final(tState st);			 // включает одно состояние
									 // во множество заключительных
	int apply(const tSymbol *input); // применяет автомат
									 // к входной цепочке

	void clear();								 // очищает автомат
	size_t size() const { return table.size(); } // выдает
												 // размер(количество состояний) автомата

	friend class tFSMcheck; // класс для проверки
							//                      корректности автомата
  private:
	// представление детерминированного конечного
	// автомата
	typedef std::map<tSymbol, tState> tTransMap;
	typedef std::vector<tTransMap> tStateTable;

	tStateTable table; //таблица состояний
	tStateSet finals;  //множество заключитеьных
					   // состояний
};
// функции-помощники
void addstr(tFSM &fsm,
			tFSM::tState from, const tFSM::tSymbol *str,
			tFSM::tState to);
void addrange(tFSM &fsm,
			  tFSM::tState from, tFSM::tSymbol first,
			  tFSM::tSymbol last, tFSM::tState to);
//------------------------------------------------------
//        –≈јЋ»«ј÷»я
inline void tFSM::add(tState from, tSymbol c, tState to){
	size_t sz = 1 +(from > to ? from : to); //1+max(from,to)
	if(sz > table.size())
		table.resize(sz); // увеличивает
						  // размер вектора до sz
	table[from][c] = to;  // два перегруженных оператора []:
						  // один для vector, другой для map.
}

inline void tFSM::final(tState st) { finals.insert(st); }

inline void tFSM::clear(){
	finals.clear();
	table.clear();
}

inline int tFSM::apply(const tSymbol *input){
	if(table.empty())
		return 0;	  // пустая таблица
					  // состояний
	tState state = 0; // начальное состояние
	int accepted = 0;

	// цикл прохода по входной цепочке
	while(*input){
		tTransMap::iterator iter;		 // итератор
										 // контейнера map
		tTransMap &trans = table[state]; // ссылка на таблицу
										 // переходов из состояния state
		if((iter = trans.find(*input)) == trans.end())
			break; // нет перехода

		state = iter->second; //новое состояние
		++accepted;
		++input;
	} //конец цикла
	  //          состояние не заключительное?
	return (finals.count(state) == 0) ? 0 : accepted;
}

inline void addstr(tFSM &fsm, tFSM::tState from, const tFSM::tSymbol *str, tFSM::tState to){
	for(; *str; ++str)
		fsm.add(from, *str, to);
}

inline void addrange(tFSM &fsm,
					 tFSM::tState from, tFSM::tSymbol first,
					 tFSM::tSymbol last, tFSM::tState to){
	for(tFSM::tSymbol i = first; i <= last; ++i)
		fsm.add(from, i, to);
}

#endif