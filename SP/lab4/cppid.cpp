//               cppid.cpp
#include <iostream>
#include <iomanip>
#include "fsm.h"
using namespace std;

int main(){
	tFSM fsm;
	///////////////////////
	//  Постоить автомат
	addstr(fsm, 0, "_", 1);
	addrange(fsm, 0, 'A', 'Z', 1);
	addrange(fsm, 0, 'a', 'z', 1);
	addstr(fsm, 1, "_", 1);
	addrange(fsm, 1, 'A', 'Z', 1);
	addrange(fsm, 1, 'a', 'z', 1);
	addrange(fsm, 1, '0', '9', 1);
	fsm.final(1);
	///////////////////////

	cout << "*** cppid  "
		 << "size=" << fsm.size()
		 << " ***\n";
	cout << endl;

	while (true){
		char input[81];
		cout << ">";
		cin.getline(input, 81);
		if (!*input)
			break;
		int res = fsm.apply(input);
		cout << setw(res ? res + 1 : 0) << "^"
			 << endl;
	}
	return 0;
}