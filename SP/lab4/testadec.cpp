//               testadec.cpp
#include <iostream>
#include <iomanip>
#include "fsm.h"
using namespace std;

int main(){
	tFSM Adec;
	///////////////////////
	//  �������� �������
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
	//......................
	///////////////////////
	cout << "*** xxx Adec "
		 << "size=" << Adec.size()
		 << " ***\n";
	cout << endl;

	while (true)
{
		char input[81];
		cout << ">";
		cin.getline(input, 81);
		if (!*input)
			break;

		// cout << "input: " << input << endl;

		int res = Adec.apply(input);
		cout << setw(res ? res + 1 : 0) << "^"
			 << endl;
	}
	return 0;
}
